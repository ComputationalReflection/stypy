
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Sane parameters for stats.distributions.
3: '''
4: 
5: distcont = [
6:     ['alpha', (3.5704770516650459,)],
7:     ['anglit', ()],
8:     ['arcsine', ()],
9:     ['argus', (1.0,)],
10:     ['beta', (2.3098496451481823, 0.62687954300963677)],
11:     ['betaprime', (5, 6)],
12:     ['bradford', (0.29891359763170633,)],
13:     ['burr', (10.5, 4.3)],
14:     ['burr12', (10, 4)],
15:     ['cauchy', ()],
16:     ['chi', (78,)],
17:     ['chi2', (55,)],
18:     ['cosine', ()],
19:     ['crystalball', (2.0, 3.0)],
20:     ['dgamma', (1.1023326088288166,)],
21:     ['dweibull', (2.0685080649914673,)],
22:     ['erlang', (10,)],
23:     ['expon', ()],
24:     ['exponnorm', (1.5,)],
25:     ['exponpow', (2.697119160358469,)],
26:     ['exponweib', (2.8923945291034436, 1.9505288745913174)],
27:     ['f', (29, 18)],
28:     ['fatiguelife', (29,)],   # correction numargs = 1
29:     ['fisk', (3.0857548622253179,)],
30:     ['foldcauchy', (4.7164673455831894,)],
31:     ['foldnorm', (1.9521253373555869,)],
32:     ['frechet_l', (3.6279911255583239,)],
33:     ['frechet_r', (1.8928171603534227,)],
34:     ['gamma', (1.9932305483800778,)],
35:     ['gausshyper', (13.763771604130699, 3.1189636648681431,
36:                     2.5145980350183019, 5.1811649903971615)],  # veryslow
37:     ['genexpon', (9.1325976465418908, 16.231956600590632, 3.2819552690843983)],
38:     ['genextreme', (-0.1,)],
39:     ['gengamma', (4.4162385429431925, 3.1193091679242761)],
40:     ['gengamma', (4.4162385429431925, -3.1193091679242761)],
41:     ['genhalflogistic', (0.77274727809929322,)],
42:     ['genlogistic', (0.41192440799679475,)],
43:     ['gennorm', (1.2988442399460265,)],
44:     ['halfgennorm', (0.6748054997000371,)],
45:     ['genpareto', (0.1,)],   # use case with finite moments
46:     ['gilbrat', ()],
47:     ['gompertz', (0.94743713075105251,)],
48:     ['gumbel_l', ()],
49:     ['gumbel_r', ()],
50:     ['halfcauchy', ()],
51:     ['halflogistic', ()],
52:     ['halfnorm', ()],
53:     ['hypsecant', ()],
54:     ['invgamma', (4.0668996136993067,)],
55:     ['invgauss', (0.14546264555347513,)],
56:     ['invweibull', (10.58,)],
57:     ['johnsonsb', (4.3172675099141058, 3.1837781130785063)],
58:     ['johnsonsu', (2.554395574161155, 2.2482281679651965)],
59:     ['kappa4', (0.0, 0.0)],
60:     ['kappa4', (-0.1, 0.1)],
61:     ['kappa4', (0.0, 0.1)],
62:     ['kappa4', (0.1, 0.0)],
63:     ['kappa3', (1.0,)],
64:     ['ksone', (1000,)],  # replace 22 by 100 to avoid failing range, ticket 956
65:     ['kstwobign', ()],
66:     ['laplace', ()],
67:     ['levy', ()],
68:     ['levy_l', ()],
69:     ['levy_stable', (0.35667405469844993,
70:                      -0.67450531578494011)],  # NotImplementedError
71:     #           rvs not tested
72:     ['loggamma', (0.41411931826052117,)],
73:     ['logistic', ()],
74:     ['loglaplace', (3.2505926592051435,)],
75:     ['lognorm', (0.95368226960575331,)],
76:     ['lomax', (1.8771398388773268,)],
77:     ['maxwell', ()],
78:     ['mielke', (10.4, 3.6)],
79:     ['nakagami', (4.9673794866666237,)],
80:     ['ncf', (27, 27, 0.41578441799226107)],
81:     ['nct', (14, 0.24045031331198066)],
82:     ['ncx2', (21, 1.0560465975116415)],
83:     ['norm', ()],
84:     ['pareto', (2.621716532144454,)],
85:     ['pearson3', (0.1,)],
86:     ['powerlaw', (1.6591133289905851,)],
87:     ['powerlognorm', (2.1413923530064087, 0.44639540782048337)],
88:     ['powernorm', (4.4453652254590779,)],
89:     ['rayleigh', ()],
90:     ['rdist', (0.9,)],   # feels also slow
91:     ['recipinvgauss', (0.63004267809369119,)],
92:     ['reciprocal', (0.0062309367010521255, 1.0062309367010522)],
93:     ['rice', (0.7749725210111873,)],
94:     ['semicircular', ()],
95:     ['skewnorm', (4.0,)],
96:     ['t', (2.7433514990818093,)],
97:     ['trapz', (0.2, 0.8)],
98:     ['triang', (0.15785029824528218,)],
99:     ['truncexpon', (4.6907725456810478,)],
100:     ['truncnorm', (-1.0978730080013919, 2.7306754109031979)],
101:     ['truncnorm', (0.1, 2.)],
102:     ['tukeylambda', (3.1321477856738267,)],
103:     ['uniform', ()],
104:     ['vonmises', (3.9939042581071398,)],
105:     ['vonmises_line', (3.9939042581071398,)],
106:     ['wald', ()],
107:     ['weibull_max', (2.8687961709100187,)],
108:     ['weibull_min', (1.7866166930421596,)],
109:     ['wrapcauchy', (0.031071279018614728,)]]
110: 
111: 
112: distdiscrete = [
113:     ['bernoulli',(0.3,)],
114:     ['binom', (5, 0.4)],
115:     ['boltzmann',(1.4, 19)],
116:     ['dlaplace', (0.8,)],  # 0.5
117:     ['geom', (0.5,)],
118:     ['hypergeom',(30, 12, 6)],
119:     ['hypergeom',(21,3,12)],  # numpy.random (3,18,12) numpy ticket:921
120:     ['hypergeom',(21,18,11)],  # numpy.random (18,3,11) numpy ticket:921
121:     ['logser', (0.6,)],  # reenabled, numpy ticket:921
122:     ['nbinom', (5, 0.5)],
123:     ['nbinom', (0.4, 0.4)],  # from tickets: 583
124:     ['planck', (0.51,)],   # 4.1
125:     ['poisson', (0.6,)],
126:     ['randint', (7, 31)],
127:     ['skellam', (15, 8)],
128:     ['zipf', (6.5,)]
129: ]
130: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_618075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nSane parameters for stats.distributions.\n')

# Assigning a List to a Name (line 5):

# Obtaining an instance of the builtin type 'list' (line 5)
list_618076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 6)
list_618077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_618078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 5), 'str', 'alpha')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 4), list_618077, str_618078)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_618079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
float_618080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 15), tuple_618079, float_618080)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 4), list_618077, tuple_618079)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618077)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 7)
list_618081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_618082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'str', 'anglit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 4), list_618081, str_618082)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_618083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 4), list_618081, tuple_618083)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618081)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 8)
list_618084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_618085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 5), 'str', 'arcsine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 4), list_618084, str_618085)
# Adding element type (line 8)

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_618086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 4), list_618084, tuple_618086)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618084)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 9)
list_618087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_618088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 5), 'str', 'argus')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_618087, str_618088)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_618089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
float_618090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), tuple_618089, float_618090)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), list_618087, tuple_618089)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618087)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 10)
list_618091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_618092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 5), 'str', 'beta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_618091, str_618092)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_618093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
float_618094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), tuple_618093, float_618094)
# Adding element type (line 10)
float_618095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), tuple_618093, float_618095)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), list_618091, tuple_618093)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618091)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 11)
list_618096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_618097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 5), 'str', 'betaprime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 4), list_618096, str_618097)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'tuple' (line 11)
tuple_618098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 11)
# Adding element type (line 11)
int_618099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 19), tuple_618098, int_618099)
# Adding element type (line 11)
int_618100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 19), tuple_618098, int_618100)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 4), list_618096, tuple_618098)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618096)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 12)
list_618101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_618102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 5), 'str', 'bradford')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_618101, str_618102)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_618103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
float_618104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), tuple_618103, float_618104)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_618101, tuple_618103)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618101)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 13)
list_618105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_618106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 5), 'str', 'burr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_618105, str_618106)
# Adding element type (line 13)

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_618107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
float_618108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 14), tuple_618107, float_618108)
# Adding element type (line 13)
float_618109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 14), tuple_618107, float_618109)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_618105, tuple_618107)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618105)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 14)
list_618110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_618111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'str', 'burr12')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_618110, str_618111)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_618112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_618113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), tuple_618112, int_618113)
# Adding element type (line 14)
int_618114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), tuple_618112, int_618114)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_618110, tuple_618112)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618110)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 15)
list_618115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_618116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'str', 'cauchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_618115, str_618116)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_618117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_618115, tuple_618117)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618115)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 16)
list_618118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_618119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'str', 'chi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_618118, str_618119)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_618120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
int_618121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), tuple_618120, int_618121)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_618118, tuple_618120)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618118)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 17)
list_618122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_618123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 5), 'str', 'chi2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), list_618122, str_618123)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_618124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
int_618125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), tuple_618124, int_618125)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 4), list_618122, tuple_618124)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618122)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 18)
list_618126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_618127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'str', 'cosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_618126, str_618127)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_618128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_618126, tuple_618128)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618126)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 19)
list_618129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_618130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 5), 'str', 'crystalball')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_618129, str_618130)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_618131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
float_618132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), tuple_618131, float_618132)
# Adding element type (line 19)
float_618133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), tuple_618131, float_618133)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_618129, tuple_618131)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618129)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 20)
list_618134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_618135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'str', 'dgamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), list_618134, str_618135)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_618136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
float_618137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), tuple_618136, float_618137)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), list_618134, tuple_618136)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618134)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 21)
list_618138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_618139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 5), 'str', 'dweibull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 4), list_618138, str_618139)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_618140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)
# Adding element type (line 21)
float_618141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 18), tuple_618140, float_618141)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 4), list_618138, tuple_618140)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618138)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 22)
list_618142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_618143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'str', 'erlang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 4), list_618142, str_618143)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_618144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
int_618145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), tuple_618144, int_618145)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 4), list_618142, tuple_618144)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618142)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 23)
list_618146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_618147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'str', 'expon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_618146, str_618147)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_618148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_618146, tuple_618148)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618146)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 24)
list_618149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_618150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'str', 'exponnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), list_618149, str_618150)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_618151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
float_618152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), tuple_618151, float_618152)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), list_618149, tuple_618151)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618149)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 25)
list_618153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_618154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'str', 'exponpow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_618153, str_618154)
# Adding element type (line 25)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_618155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
float_618156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), tuple_618155, float_618156)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), list_618153, tuple_618155)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618153)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 26)
list_618157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_618158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'str', 'exponweib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_618157, str_618158)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_618159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
float_618160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), tuple_618159, float_618160)
# Adding element type (line 26)
float_618161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), tuple_618159, float_618161)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), list_618157, tuple_618159)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618157)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 27)
list_618162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_618163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_618162, str_618163)
# Adding element type (line 27)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_618164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
int_618165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 11), tuple_618164, int_618165)
# Adding element type (line 27)
int_618166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 11), tuple_618164, int_618166)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_618162, tuple_618164)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618162)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 28)
list_618167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_618168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 5), 'str', 'fatiguelife')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), list_618167, str_618168)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_618169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
int_618170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_618169, int_618170)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), list_618167, tuple_618169)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618167)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 29)
list_618171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_618172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'str', 'fisk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), list_618171, str_618172)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_618173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
float_618174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), tuple_618173, float_618174)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), list_618171, tuple_618173)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618171)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 30)
list_618175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
str_618176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 5), 'str', 'foldcauchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_618175, str_618176)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_618177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
float_618178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 20), tuple_618177, float_618178)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), list_618175, tuple_618177)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618175)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 31)
list_618179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
str_618180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 5), 'str', 'foldnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_618179, str_618180)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_618181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
float_618182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 18), tuple_618181, float_618182)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_618179, tuple_618181)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618179)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 32)
list_618183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
str_618184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 5), 'str', 'frechet_l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), list_618183, str_618184)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_618185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
float_618186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 19), tuple_618185, float_618186)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), list_618183, tuple_618185)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618183)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 33)
list_618187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
str_618188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 5), 'str', 'frechet_r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_618187, str_618188)
# Adding element type (line 33)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_618189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
float_618190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), tuple_618189, float_618190)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_618187, tuple_618189)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618187)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 34)
list_618191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
str_618192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 5), 'str', 'gamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), list_618191, str_618192)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_618193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)
float_618194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 15), tuple_618193, float_618194)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), list_618191, tuple_618193)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618191)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 35)
list_618195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)
# Adding element type (line 35)
str_618196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 5), 'str', 'gausshyper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 4), list_618195, str_618196)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_618197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
float_618198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), tuple_618197, float_618198)
# Adding element type (line 35)
float_618199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), tuple_618197, float_618199)
# Adding element type (line 35)
float_618200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), tuple_618197, float_618200)
# Adding element type (line 35)
float_618201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), tuple_618197, float_618201)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 4), list_618195, tuple_618197)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618195)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 37)
list_618202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
str_618203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 5), 'str', 'genexpon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), list_618202, str_618203)
# Adding element type (line 37)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_618204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
float_618205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), tuple_618204, float_618205)
# Adding element type (line 37)
float_618206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), tuple_618204, float_618206)
# Adding element type (line 37)
float_618207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 58), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), tuple_618204, float_618207)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), list_618202, tuple_618204)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618202)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 38)
list_618208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
str_618209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 5), 'str', 'genextreme')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 4), list_618208, str_618209)
# Adding element type (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 38)
tuple_618210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 38)
# Adding element type (line 38)
float_618211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_618210, float_618211)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 4), list_618208, tuple_618210)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618208)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 39)
list_618212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
str_618213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 5), 'str', 'gengamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), list_618212, str_618213)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_618214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
float_618215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), tuple_618214, float_618215)
# Adding element type (line 39)
float_618216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), tuple_618214, float_618216)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), list_618212, tuple_618214)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618212)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 40)
list_618217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
str_618218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 5), 'str', 'gengamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_618217, str_618218)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_618219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
float_618220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), tuple_618219, float_618220)
# Adding element type (line 40)
float_618221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 18), tuple_618219, float_618221)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_618217, tuple_618219)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618217)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 41)
list_618222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
str_618223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 5), 'str', 'genhalflogistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 4), list_618222, str_618223)
# Adding element type (line 41)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_618224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
float_618225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 25), tuple_618224, float_618225)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 4), list_618222, tuple_618224)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618222)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 42)
list_618226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)
str_618227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 5), 'str', 'genlogistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), list_618226, str_618227)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_618228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
float_618229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_618228, float_618229)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), list_618226, tuple_618228)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618226)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 43)
list_618230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
str_618231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 5), 'str', 'gennorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), list_618230, str_618231)
# Adding element type (line 43)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_618232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
float_618233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 17), tuple_618232, float_618233)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), list_618230, tuple_618232)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618230)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 44)
list_618234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)
# Adding element type (line 44)
str_618235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 5), 'str', 'halfgennorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), list_618234, str_618235)
# Adding element type (line 44)

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_618236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
float_618237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_618236, float_618237)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), list_618234, tuple_618236)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618234)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 45)
list_618238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 45)
# Adding element type (line 45)
str_618239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 5), 'str', 'genpareto')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), list_618238, str_618239)
# Adding element type (line 45)

# Obtaining an instance of the builtin type 'tuple' (line 45)
tuple_618240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 45)
# Adding element type (line 45)
float_618241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_618240, float_618241)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), list_618238, tuple_618240)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618238)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 46)
list_618242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 46)
# Adding element type (line 46)
str_618243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 5), 'str', 'gilbrat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 4), list_618242, str_618243)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_618244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 4), list_618242, tuple_618244)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618242)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 47)
list_618245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
str_618246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 5), 'str', 'gompertz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), list_618245, str_618246)
# Adding element type (line 47)

# Obtaining an instance of the builtin type 'tuple' (line 47)
tuple_618247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 47)
# Adding element type (line 47)
float_618248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 18), tuple_618247, float_618248)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), list_618245, tuple_618247)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618245)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 48)
list_618249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
str_618250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 5), 'str', 'gumbel_l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), list_618249, str_618250)
# Adding element type (line 48)

# Obtaining an instance of the builtin type 'tuple' (line 48)
tuple_618251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 48)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), list_618249, tuple_618251)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618249)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 49)
list_618252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)
str_618253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 5), 'str', 'gumbel_r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 4), list_618252, str_618253)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 49)
tuple_618254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 49)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 4), list_618252, tuple_618254)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618252)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 50)
list_618255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)
str_618256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 5), 'str', 'halfcauchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 4), list_618255, str_618256)
# Adding element type (line 50)

# Obtaining an instance of the builtin type 'tuple' (line 50)
tuple_618257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 50)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 4), list_618255, tuple_618257)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618255)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 51)
list_618258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 51)
# Adding element type (line 51)
str_618259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 5), 'str', 'halflogistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 4), list_618258, str_618259)
# Adding element type (line 51)

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_618260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 4), list_618258, tuple_618260)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618258)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 52)
list_618261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 52)
# Adding element type (line 52)
str_618262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 5), 'str', 'halfnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), list_618261, str_618262)
# Adding element type (line 52)

# Obtaining an instance of the builtin type 'tuple' (line 52)
tuple_618263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 52)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), list_618261, tuple_618263)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618261)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 53)
list_618264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 53)
# Adding element type (line 53)
str_618265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 5), 'str', 'hypsecant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 4), list_618264, str_618265)
# Adding element type (line 53)

# Obtaining an instance of the builtin type 'tuple' (line 53)
tuple_618266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 53)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 4), list_618264, tuple_618266)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618264)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 54)
list_618267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
str_618268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 5), 'str', 'invgamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 4), list_618267, str_618268)
# Adding element type (line 54)

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_618269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
float_618270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), tuple_618269, float_618270)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 4), list_618267, tuple_618269)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618267)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 55)
list_618271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
str_618272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 5), 'str', 'invgauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 4), list_618271, str_618272)
# Adding element type (line 55)

# Obtaining an instance of the builtin type 'tuple' (line 55)
tuple_618273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 55)
# Adding element type (line 55)
float_618274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), tuple_618273, float_618274)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 4), list_618271, tuple_618273)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618271)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 56)
list_618275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 56)
# Adding element type (line 56)
str_618276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 5), 'str', 'invweibull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), list_618275, str_618276)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_618277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
float_618278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), tuple_618277, float_618278)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), list_618275, tuple_618277)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618275)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 57)
list_618279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
str_618280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 5), 'str', 'johnsonsb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 4), list_618279, str_618280)
# Adding element type (line 57)

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_618281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
float_618282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_618281, float_618282)
# Adding element type (line 57)
float_618283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_618281, float_618283)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 4), list_618279, tuple_618281)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618279)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 58)
list_618284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 58)
# Adding element type (line 58)
str_618285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 5), 'str', 'johnsonsu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 4), list_618284, str_618285)
# Adding element type (line 58)

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_618286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)
float_618287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 19), tuple_618286, float_618287)
# Adding element type (line 58)
float_618288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 19), tuple_618286, float_618288)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 4), list_618284, tuple_618286)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618284)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 59)
list_618289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 59)
# Adding element type (line 59)
str_618290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 5), 'str', 'kappa4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), list_618289, str_618290)
# Adding element type (line 59)

# Obtaining an instance of the builtin type 'tuple' (line 59)
tuple_618291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 59)
# Adding element type (line 59)
float_618292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), tuple_618291, float_618292)
# Adding element type (line 59)
float_618293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), tuple_618291, float_618293)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), list_618289, tuple_618291)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618289)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 60)
list_618294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
str_618295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 5), 'str', 'kappa4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), list_618294, str_618295)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_618296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)
float_618297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 16), tuple_618296, float_618297)
# Adding element type (line 60)
float_618298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 16), tuple_618296, float_618298)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), list_618294, tuple_618296)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618294)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 61)
list_618299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
str_618300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 5), 'str', 'kappa4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 4), list_618299, str_618300)
# Adding element type (line 61)

# Obtaining an instance of the builtin type 'tuple' (line 61)
tuple_618301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 61)
# Adding element type (line 61)
float_618302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), tuple_618301, float_618302)
# Adding element type (line 61)
float_618303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), tuple_618301, float_618303)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 4), list_618299, tuple_618301)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618299)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 62)
list_618304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
str_618305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 5), 'str', 'kappa4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 4), list_618304, str_618305)
# Adding element type (line 62)

# Obtaining an instance of the builtin type 'tuple' (line 62)
tuple_618306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 62)
# Adding element type (line 62)
float_618307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), tuple_618306, float_618307)
# Adding element type (line 62)
float_618308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 16), tuple_618306, float_618308)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 4), list_618304, tuple_618306)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618304)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 63)
list_618309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
str_618310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 5), 'str', 'kappa3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 4), list_618309, str_618310)
# Adding element type (line 63)

# Obtaining an instance of the builtin type 'tuple' (line 63)
tuple_618311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 63)
# Adding element type (line 63)
float_618312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 16), tuple_618311, float_618312)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 4), list_618309, tuple_618311)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618309)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 64)
list_618313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 64)
# Adding element type (line 64)
str_618314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 5), 'str', 'ksone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), list_618313, str_618314)
# Adding element type (line 64)

# Obtaining an instance of the builtin type 'tuple' (line 64)
tuple_618315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 64)
# Adding element type (line 64)
int_618316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 15), tuple_618315, int_618316)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), list_618313, tuple_618315)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618313)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 65)
list_618317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 65)
# Adding element type (line 65)
str_618318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 5), 'str', 'kstwobign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 4), list_618317, str_618318)
# Adding element type (line 65)

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_618319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 4), list_618317, tuple_618319)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618317)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 66)
list_618320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 66)
# Adding element type (line 66)
str_618321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 5), 'str', 'laplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 4), list_618320, str_618321)
# Adding element type (line 66)

# Obtaining an instance of the builtin type 'tuple' (line 66)
tuple_618322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 66)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 4), list_618320, tuple_618322)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618320)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 67)
list_618323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 67)
# Adding element type (line 67)
str_618324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 5), 'str', 'levy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 4), list_618323, str_618324)
# Adding element type (line 67)

# Obtaining an instance of the builtin type 'tuple' (line 67)
tuple_618325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 67)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 4), list_618323, tuple_618325)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618323)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 68)
list_618326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 68)
# Adding element type (line 68)
str_618327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 5), 'str', 'levy_l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), list_618326, str_618327)
# Adding element type (line 68)

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_618328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), list_618326, tuple_618328)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618326)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 69)
list_618329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
str_618330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 5), 'str', 'levy_stable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), list_618329, str_618330)
# Adding element type (line 69)

# Obtaining an instance of the builtin type 'tuple' (line 69)
tuple_618331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 69)
# Adding element type (line 69)
float_618332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 21), tuple_618331, float_618332)
# Adding element type (line 69)
float_618333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 21), tuple_618331, float_618333)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), list_618329, tuple_618331)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618329)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 72)
list_618334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 72)
# Adding element type (line 72)
str_618335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 5), 'str', 'loggamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), list_618334, str_618335)
# Adding element type (line 72)

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_618336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
float_618337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), tuple_618336, float_618337)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 4), list_618334, tuple_618336)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618334)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 73)
list_618338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 73)
# Adding element type (line 73)
str_618339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 5), 'str', 'logistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 4), list_618338, str_618339)
# Adding element type (line 73)

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_618340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 4), list_618338, tuple_618340)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618338)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 74)
list_618341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 74)
# Adding element type (line 74)
str_618342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 5), 'str', 'loglaplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 4), list_618341, str_618342)
# Adding element type (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_618343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
float_618344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_618343, float_618344)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 4), list_618341, tuple_618343)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618341)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 75)
list_618345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 75)
# Adding element type (line 75)
str_618346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 5), 'str', 'lognorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 4), list_618345, str_618346)
# Adding element type (line 75)

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_618347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
float_618348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), tuple_618347, float_618348)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 4), list_618345, tuple_618347)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618345)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 76)
list_618349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 76)
# Adding element type (line 76)
str_618350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 5), 'str', 'lomax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 4), list_618349, str_618350)
# Adding element type (line 76)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_618351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
float_618352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), tuple_618351, float_618352)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 4), list_618349, tuple_618351)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618349)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 77)
list_618353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 77)
# Adding element type (line 77)
str_618354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 5), 'str', 'maxwell')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 4), list_618353, str_618354)
# Adding element type (line 77)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_618355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 4), list_618353, tuple_618355)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618353)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 78)
list_618356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 78)
# Adding element type (line 78)
str_618357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 5), 'str', 'mielke')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 4), list_618356, str_618357)
# Adding element type (line 78)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_618358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
float_618359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 16), tuple_618358, float_618359)
# Adding element type (line 78)
float_618360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 16), tuple_618358, float_618360)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 4), list_618356, tuple_618358)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618356)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 79)
list_618361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 79)
# Adding element type (line 79)
str_618362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 5), 'str', 'nakagami')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 4), list_618361, str_618362)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_618363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
float_618364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 18), tuple_618363, float_618364)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 4), list_618361, tuple_618363)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618361)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 80)
list_618365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 80)
# Adding element type (line 80)
str_618366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 5), 'str', 'ncf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 4), list_618365, str_618366)
# Adding element type (line 80)

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_618367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
int_618368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), tuple_618367, int_618368)
# Adding element type (line 80)
int_618369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), tuple_618367, int_618369)
# Adding element type (line 80)
float_618370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), tuple_618367, float_618370)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 4), list_618365, tuple_618367)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618365)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 81)
list_618371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 81)
# Adding element type (line 81)
str_618372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 5), 'str', 'nct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 4), list_618371, str_618372)
# Adding element type (line 81)

# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_618373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)
int_618374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 13), tuple_618373, int_618374)
# Adding element type (line 81)
float_618375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 13), tuple_618373, float_618375)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 4), list_618371, tuple_618373)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618371)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 82)
list_618376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 82)
# Adding element type (line 82)
str_618377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 5), 'str', 'ncx2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 4), list_618376, str_618377)
# Adding element type (line 82)

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_618378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
int_618379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 14), tuple_618378, int_618379)
# Adding element type (line 82)
float_618380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 14), tuple_618378, float_618380)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 4), list_618376, tuple_618378)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618376)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 83)
list_618381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 83)
# Adding element type (line 83)
str_618382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 5), 'str', 'norm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 4), list_618381, str_618382)
# Adding element type (line 83)

# Obtaining an instance of the builtin type 'tuple' (line 83)
tuple_618383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 83)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 4), list_618381, tuple_618383)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618381)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 84)
list_618384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 84)
# Adding element type (line 84)
str_618385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 5), 'str', 'pareto')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 4), list_618384, str_618385)
# Adding element type (line 84)

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_618386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
float_618387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_618386, float_618387)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 4), list_618384, tuple_618386)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618384)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 85)
list_618388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 85)
# Adding element type (line 85)
str_618389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 5), 'str', 'pearson3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), list_618388, str_618389)
# Adding element type (line 85)

# Obtaining an instance of the builtin type 'tuple' (line 85)
tuple_618390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 85)
# Adding element type (line 85)
float_618391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), tuple_618390, float_618391)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 4), list_618388, tuple_618390)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618388)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 86)
list_618392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 86)
# Adding element type (line 86)
str_618393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 5), 'str', 'powerlaw')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 4), list_618392, str_618393)
# Adding element type (line 86)

# Obtaining an instance of the builtin type 'tuple' (line 86)
tuple_618394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 86)
# Adding element type (line 86)
float_618395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), tuple_618394, float_618395)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 4), list_618392, tuple_618394)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618392)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 87)
list_618396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 87)
# Adding element type (line 87)
str_618397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 5), 'str', 'powerlognorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 4), list_618396, str_618397)
# Adding element type (line 87)

# Obtaining an instance of the builtin type 'tuple' (line 87)
tuple_618398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 87)
# Adding element type (line 87)
float_618399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 22), tuple_618398, float_618399)
# Adding element type (line 87)
float_618400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 22), tuple_618398, float_618400)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 4), list_618396, tuple_618398)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618396)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 88)
list_618401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 88)
# Adding element type (line 88)
str_618402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 5), 'str', 'powernorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 4), list_618401, str_618402)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 88)
tuple_618403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 88)
# Adding element type (line 88)
float_618404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), tuple_618403, float_618404)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 4), list_618401, tuple_618403)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618401)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 89)
list_618405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 89)
# Adding element type (line 89)
str_618406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 5), 'str', 'rayleigh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 4), list_618405, str_618406)
# Adding element type (line 89)

# Obtaining an instance of the builtin type 'tuple' (line 89)
tuple_618407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 89)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 4), list_618405, tuple_618407)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618405)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 90)
list_618408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 90)
# Adding element type (line 90)
str_618409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 5), 'str', 'rdist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 4), list_618408, str_618409)
# Adding element type (line 90)

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_618410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
float_618411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), tuple_618410, float_618411)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 4), list_618408, tuple_618410)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618408)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 91)
list_618412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 91)
# Adding element type (line 91)
str_618413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 5), 'str', 'recipinvgauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 4), list_618412, str_618413)
# Adding element type (line 91)

# Obtaining an instance of the builtin type 'tuple' (line 91)
tuple_618414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 91)
# Adding element type (line 91)
float_618415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 23), tuple_618414, float_618415)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 4), list_618412, tuple_618414)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618412)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 92)
list_618416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 92)
# Adding element type (line 92)
str_618417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 5), 'str', 'reciprocal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), list_618416, str_618417)
# Adding element type (line 92)

# Obtaining an instance of the builtin type 'tuple' (line 92)
tuple_618418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 92)
# Adding element type (line 92)
float_618419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 20), tuple_618418, float_618419)
# Adding element type (line 92)
float_618420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 20), tuple_618418, float_618420)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), list_618416, tuple_618418)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618416)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 93)
list_618421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 93)
# Adding element type (line 93)
str_618422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 5), 'str', 'rice')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 4), list_618421, str_618422)
# Adding element type (line 93)

# Obtaining an instance of the builtin type 'tuple' (line 93)
tuple_618423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 93)
# Adding element type (line 93)
float_618424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 14), tuple_618423, float_618424)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 4), list_618421, tuple_618423)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618421)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 94)
list_618425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 94)
# Adding element type (line 94)
str_618426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 5), 'str', 'semicircular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 4), list_618425, str_618426)
# Adding element type (line 94)

# Obtaining an instance of the builtin type 'tuple' (line 94)
tuple_618427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 94)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 4), list_618425, tuple_618427)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618425)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 95)
list_618428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 95)
# Adding element type (line 95)
str_618429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 5), 'str', 'skewnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 4), list_618428, str_618429)
# Adding element type (line 95)

# Obtaining an instance of the builtin type 'tuple' (line 95)
tuple_618430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 95)
# Adding element type (line 95)
float_618431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 18), tuple_618430, float_618431)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 4), list_618428, tuple_618430)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618428)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 96)
list_618432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 96)
# Adding element type (line 96)
str_618433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 5), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 4), list_618432, str_618433)
# Adding element type (line 96)

# Obtaining an instance of the builtin type 'tuple' (line 96)
tuple_618434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 96)
# Adding element type (line 96)
float_618435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 11), tuple_618434, float_618435)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 4), list_618432, tuple_618434)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618432)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 97)
list_618436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 97)
# Adding element type (line 97)
str_618437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 5), 'str', 'trapz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 4), list_618436, str_618437)
# Adding element type (line 97)

# Obtaining an instance of the builtin type 'tuple' (line 97)
tuple_618438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 97)
# Adding element type (line 97)
float_618439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 15), tuple_618438, float_618439)
# Adding element type (line 97)
float_618440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 15), tuple_618438, float_618440)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 4), list_618436, tuple_618438)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618436)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 98)
list_618441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 98)
# Adding element type (line 98)
str_618442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 5), 'str', 'triang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 4), list_618441, str_618442)
# Adding element type (line 98)

# Obtaining an instance of the builtin type 'tuple' (line 98)
tuple_618443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 98)
# Adding element type (line 98)
float_618444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), tuple_618443, float_618444)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 4), list_618441, tuple_618443)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618441)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 99)
list_618445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 99)
# Adding element type (line 99)
str_618446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 5), 'str', 'truncexpon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 4), list_618445, str_618446)
# Adding element type (line 99)

# Obtaining an instance of the builtin type 'tuple' (line 99)
tuple_618447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 99)
# Adding element type (line 99)
float_618448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 20), tuple_618447, float_618448)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 4), list_618445, tuple_618447)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618445)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 100)
list_618449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 100)
# Adding element type (line 100)
str_618450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 5), 'str', 'truncnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 4), list_618449, str_618450)
# Adding element type (line 100)

# Obtaining an instance of the builtin type 'tuple' (line 100)
tuple_618451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 100)
# Adding element type (line 100)
float_618452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 19), tuple_618451, float_618452)
# Adding element type (line 100)
float_618453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 19), tuple_618451, float_618453)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 4), list_618449, tuple_618451)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618449)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 101)
list_618454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)
# Adding element type (line 101)
str_618455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 5), 'str', 'truncnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 4), list_618454, str_618455)
# Adding element type (line 101)

# Obtaining an instance of the builtin type 'tuple' (line 101)
tuple_618456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 101)
# Adding element type (line 101)
float_618457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_618456, float_618457)
# Adding element type (line 101)
float_618458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_618456, float_618458)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 4), list_618454, tuple_618456)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618454)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 102)
list_618459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 102)
# Adding element type (line 102)
str_618460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 5), 'str', 'tukeylambda')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 4), list_618459, str_618460)
# Adding element type (line 102)

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_618461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
float_618462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 21), tuple_618461, float_618462)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 4), list_618459, tuple_618461)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618459)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 103)
list_618463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 103)
# Adding element type (line 103)
str_618464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 5), 'str', 'uniform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 4), list_618463, str_618464)
# Adding element type (line 103)

# Obtaining an instance of the builtin type 'tuple' (line 103)
tuple_618465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 103)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 4), list_618463, tuple_618465)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618463)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 104)
list_618466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 104)
# Adding element type (line 104)
str_618467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 5), 'str', 'vonmises')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 4), list_618466, str_618467)
# Adding element type (line 104)

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_618468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
float_618469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 18), tuple_618468, float_618469)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 4), list_618466, tuple_618468)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618466)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 105)
list_618470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 105)
# Adding element type (line 105)
str_618471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 5), 'str', 'vonmises_line')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 4), list_618470, str_618471)
# Adding element type (line 105)

# Obtaining an instance of the builtin type 'tuple' (line 105)
tuple_618472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 105)
# Adding element type (line 105)
float_618473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_618472, float_618473)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 4), list_618470, tuple_618472)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618470)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 106)
list_618474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 106)
# Adding element type (line 106)
str_618475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 5), 'str', 'wald')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 4), list_618474, str_618475)
# Adding element type (line 106)

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_618476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 4), list_618474, tuple_618476)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618474)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 107)
list_618477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 107)
# Adding element type (line 107)
str_618478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 5), 'str', 'weibull_max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 4), list_618477, str_618478)
# Adding element type (line 107)

# Obtaining an instance of the builtin type 'tuple' (line 107)
tuple_618479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 107)
# Adding element type (line 107)
float_618480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), tuple_618479, float_618480)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 4), list_618477, tuple_618479)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618477)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 108)
list_618481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 108)
# Adding element type (line 108)
str_618482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 5), 'str', 'weibull_min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 4), list_618481, str_618482)
# Adding element type (line 108)

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_618483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
float_618484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), tuple_618483, float_618484)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 4), list_618481, tuple_618483)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618481)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 109)
list_618485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 109)
# Adding element type (line 109)
str_618486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 5), 'str', 'wrapcauchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 4), list_618485, str_618486)
# Adding element type (line 109)

# Obtaining an instance of the builtin type 'tuple' (line 109)
tuple_618487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 109)
# Adding element type (line 109)
float_618488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), tuple_618487, float_618488)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 4), list_618485, tuple_618487)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 11), list_618076, list_618485)

# Assigning a type to the variable 'distcont' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distcont', list_618076)

# Assigning a List to a Name (line 112):

# Obtaining an instance of the builtin type 'list' (line 112)
list_618489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 112)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 113)
list_618490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 113)
# Adding element type (line 113)
str_618491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 5), 'str', 'bernoulli')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 4), list_618490, str_618491)
# Adding element type (line 113)

# Obtaining an instance of the builtin type 'tuple' (line 113)
tuple_618492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 113)
# Adding element type (line 113)
float_618493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 18), tuple_618492, float_618493)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 4), list_618490, tuple_618492)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618490)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 114)
list_618494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 114)
# Adding element type (line 114)
str_618495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 5), 'str', 'binom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 4), list_618494, str_618495)
# Adding element type (line 114)

# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_618496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
int_618497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 15), tuple_618496, int_618497)
# Adding element type (line 114)
float_618498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 15), tuple_618496, float_618498)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 4), list_618494, tuple_618496)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618494)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 115)
list_618499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 115)
# Adding element type (line 115)
str_618500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 5), 'str', 'boltzmann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 4), list_618499, str_618500)
# Adding element type (line 115)

# Obtaining an instance of the builtin type 'tuple' (line 115)
tuple_618501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 115)
# Adding element type (line 115)
float_618502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 18), tuple_618501, float_618502)
# Adding element type (line 115)
int_618503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 18), tuple_618501, int_618503)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 4), list_618499, tuple_618501)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618499)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 116)
list_618504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 116)
# Adding element type (line 116)
str_618505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 5), 'str', 'dlaplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 4), list_618504, str_618505)
# Adding element type (line 116)

# Obtaining an instance of the builtin type 'tuple' (line 116)
tuple_618506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 116)
# Adding element type (line 116)
float_618507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 18), tuple_618506, float_618507)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 4), list_618504, tuple_618506)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618504)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 117)
list_618508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 117)
# Adding element type (line 117)
str_618509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 5), 'str', 'geom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 4), list_618508, str_618509)
# Adding element type (line 117)

# Obtaining an instance of the builtin type 'tuple' (line 117)
tuple_618510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 117)
# Adding element type (line 117)
float_618511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 14), tuple_618510, float_618511)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 4), list_618508, tuple_618510)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618508)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 118)
list_618512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 118)
# Adding element type (line 118)
str_618513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 5), 'str', 'hypergeom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 4), list_618512, str_618513)
# Adding element type (line 118)

# Obtaining an instance of the builtin type 'tuple' (line 118)
tuple_618514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 118)
# Adding element type (line 118)
int_618515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_618514, int_618515)
# Adding element type (line 118)
int_618516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_618514, int_618516)
# Adding element type (line 118)
int_618517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_618514, int_618517)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 4), list_618512, tuple_618514)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618512)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 119)
list_618518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 119)
# Adding element type (line 119)
str_618519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 5), 'str', 'hypergeom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 4), list_618518, str_618519)
# Adding element type (line 119)

# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_618520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
int_618521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_618520, int_618521)
# Adding element type (line 119)
int_618522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_618520, int_618522)
# Adding element type (line 119)
int_618523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_618520, int_618523)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 4), list_618518, tuple_618520)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618518)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 120)
list_618524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 120)
# Adding element type (line 120)
str_618525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 5), 'str', 'hypergeom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 4), list_618524, str_618525)
# Adding element type (line 120)

# Obtaining an instance of the builtin type 'tuple' (line 120)
tuple_618526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 120)
# Adding element type (line 120)
int_618527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_618526, int_618527)
# Adding element type (line 120)
int_618528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_618526, int_618528)
# Adding element type (line 120)
int_618529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_618526, int_618529)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 4), list_618524, tuple_618526)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618524)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 121)
list_618530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 121)
# Adding element type (line 121)
str_618531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 5), 'str', 'logser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 4), list_618530, str_618531)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_618532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
float_618533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 16), tuple_618532, float_618533)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 4), list_618530, tuple_618532)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618530)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 122)
list_618534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 122)
# Adding element type (line 122)
str_618535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 5), 'str', 'nbinom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), list_618534, str_618535)
# Adding element type (line 122)

# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_618536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
int_618537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), tuple_618536, int_618537)
# Adding element type (line 122)
float_618538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), tuple_618536, float_618538)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 4), list_618534, tuple_618536)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618534)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 123)
list_618539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 123)
# Adding element type (line 123)
str_618540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 5), 'str', 'nbinom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), list_618539, str_618540)
# Adding element type (line 123)

# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_618541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
float_618542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_618541, float_618542)
# Adding element type (line 123)
float_618543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), tuple_618541, float_618543)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), list_618539, tuple_618541)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618539)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 124)
list_618544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 124)
# Adding element type (line 124)
str_618545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 5), 'str', 'planck')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 4), list_618544, str_618545)
# Adding element type (line 124)

# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_618546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
float_618547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), tuple_618546, float_618547)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 4), list_618544, tuple_618546)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618544)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 125)
list_618548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 125)
# Adding element type (line 125)
str_618549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 5), 'str', 'poisson')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 4), list_618548, str_618549)
# Adding element type (line 125)

# Obtaining an instance of the builtin type 'tuple' (line 125)
tuple_618550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 125)
# Adding element type (line 125)
float_618551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), tuple_618550, float_618551)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 4), list_618548, tuple_618550)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618548)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 126)
list_618552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 126)
# Adding element type (line 126)
str_618553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 5), 'str', 'randint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 4), list_618552, str_618553)
# Adding element type (line 126)

# Obtaining an instance of the builtin type 'tuple' (line 126)
tuple_618554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 126)
# Adding element type (line 126)
int_618555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), tuple_618554, int_618555)
# Adding element type (line 126)
int_618556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), tuple_618554, int_618556)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 4), list_618552, tuple_618554)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618552)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 127)
list_618557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 127)
# Adding element type (line 127)
str_618558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 5), 'str', 'skellam')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 4), list_618557, str_618558)
# Adding element type (line 127)

# Obtaining an instance of the builtin type 'tuple' (line 127)
tuple_618559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 127)
# Adding element type (line 127)
int_618560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), tuple_618559, int_618560)
# Adding element type (line 127)
int_618561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), tuple_618559, int_618561)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 4), list_618557, tuple_618559)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618557)
# Adding element type (line 112)

# Obtaining an instance of the builtin type 'list' (line 128)
list_618562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 128)
# Adding element type (line 128)
str_618563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 5), 'str', 'zipf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 4), list_618562, str_618563)
# Adding element type (line 128)

# Obtaining an instance of the builtin type 'tuple' (line 128)
tuple_618564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 128)
# Adding element type (line 128)
float_618565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 14), tuple_618564, float_618565)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 4), list_618562, tuple_618564)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_618489, list_618562)

# Assigning a type to the variable 'distdiscrete' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'distdiscrete', list_618489)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
