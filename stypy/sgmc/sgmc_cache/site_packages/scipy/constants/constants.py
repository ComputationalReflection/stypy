
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Collection of physical constants and conversion factors.
3: 
4: Most constants are in SI units, so you can do
5: print '10 mile per minute is', 10*mile/minute, 'm/s or', 10*mile/(minute*knot), 'knots'
6: 
7: The list is not meant to be comprehensive, but just a convenient list for everyday use.
8: '''
9: from __future__ import division, print_function, absolute_import
10: 
11: '''
12: BasSw 2006
13: physical constants: imported from CODATA
14: unit conversion: see e.g. NIST special publication 811
15: Use at own risk: double-check values before calculating your Mars orbit-insertion burn.
16: Some constants exist in a few variants, which are marked with suffixes.
17: The ones without any suffix should be the most common one.
18: '''
19: 
20: import math as _math
21: from .codata import value as _cd
22: import numpy as _np
23: 
24: # mathematical constants
25: pi = _math.pi
26: golden = golden_ratio = (1 + _math.sqrt(5)) / 2
27: 
28: # SI prefixes
29: yotta = 1e24
30: zetta = 1e21
31: exa = 1e18
32: peta = 1e15
33: tera = 1e12
34: giga = 1e9
35: mega = 1e6
36: kilo = 1e3
37: hecto = 1e2
38: deka = 1e1
39: deci = 1e-1
40: centi = 1e-2
41: milli = 1e-3
42: micro = 1e-6
43: nano = 1e-9
44: pico = 1e-12
45: femto = 1e-15
46: atto = 1e-18
47: zepto = 1e-21
48: 
49: # binary prefixes
50: kibi = 2**10
51: mebi = 2**20
52: gibi = 2**30
53: tebi = 2**40
54: pebi = 2**50
55: exbi = 2**60
56: zebi = 2**70
57: yobi = 2**80
58: 
59: # physical constants
60: c = speed_of_light = _cd('speed of light in vacuum')
61: mu_0 = 4e-7*pi
62: epsilon_0 = 1 / (mu_0*c*c)
63: h = Planck = _cd('Planck constant')
64: hbar = h / (2 * pi)
65: G = gravitational_constant = _cd('Newtonian constant of gravitation')
66: g = _cd('standard acceleration of gravity')
67: e = elementary_charge = _cd('elementary charge')
68: R = gas_constant = _cd('molar gas constant')
69: alpha = fine_structure = _cd('fine-structure constant')
70: N_A = Avogadro = _cd('Avogadro constant')
71: k = Boltzmann = _cd('Boltzmann constant')
72: sigma = Stefan_Boltzmann = _cd('Stefan-Boltzmann constant')
73: Wien = _cd('Wien wavelength displacement law constant')
74: Rydberg = _cd('Rydberg constant')
75: 
76: # mass in kg
77: gram = 1e-3
78: metric_ton = 1e3
79: grain = 64.79891e-6
80: lb = pound = 7000 * grain  # avoirdupois
81: blob = slinch = pound * g / 0.0254  # lbf*s**2/in (added in 1.0.0)
82: slug = blob / 12  # lbf*s**2/foot (added in 1.0.0)
83: oz = ounce = pound / 16
84: stone = 14 * pound
85: long_ton = 2240 * pound
86: short_ton = 2000 * pound
87: 
88: troy_ounce = 480 * grain  # only for metals / gems
89: troy_pound = 12 * troy_ounce
90: carat = 200e-6
91: 
92: m_e = electron_mass = _cd('electron mass')
93: m_p = proton_mass = _cd('proton mass')
94: m_n = neutron_mass = _cd('neutron mass')
95: m_u = u = atomic_mass = _cd('atomic mass constant')
96: 
97: # angle in rad
98: degree = pi / 180
99: arcmin = arcminute = degree / 60
100: arcsec = arcsecond = arcmin / 60
101: 
102: # time in second
103: minute = 60.0
104: hour = 60 * minute
105: day = 24 * hour
106: week = 7 * day
107: year = 365 * day
108: Julian_year = 365.25 * day
109: 
110: # length in meter
111: inch = 0.0254
112: foot = 12 * inch
113: yard = 3 * foot
114: mile = 1760 * yard
115: mil = inch / 1000
116: pt = point = inch / 72  # typography
117: survey_foot = 1200.0 / 3937
118: survey_mile = 5280 * survey_foot
119: nautical_mile = 1852.0
120: fermi = 1e-15
121: angstrom = 1e-10
122: micron = 1e-6
123: au = astronomical_unit = 149597870691.0
124: light_year = Julian_year * c
125: parsec = au / arcsec
126: 
127: # pressure in pascal
128: atm = atmosphere = _cd('standard atmosphere')
129: bar = 1e5
130: torr = mmHg = atm / 760
131: psi = pound * g / (inch * inch)
132: 
133: # area in meter**2
134: hectare = 1e4
135: acre = 43560 * foot**2
136: 
137: # volume in meter**3
138: litre = liter = 1e-3
139: gallon = gallon_US = 231 * inch**3  # US
140: # pint = gallon_US / 8
141: fluid_ounce = fluid_ounce_US = gallon_US / 128
142: bbl = barrel = 42 * gallon_US  # for oil
143: 
144: gallon_imp = 4.54609e-3  # UK
145: fluid_ounce_imp = gallon_imp / 160
146: 
147: # speed in meter per second
148: kmh = 1e3 / hour
149: mph = mile / hour
150: mach = speed_of_sound = 340.5  # approx value at 15 degrees in 1 atm. is this a common value?
151: knot = nautical_mile / hour
152: 
153: # temperature in kelvin
154: zero_Celsius = 273.15
155: degree_Fahrenheit = 1/1.8  # only for differences
156: 
157: # energy in joule
158: eV = electron_volt = elementary_charge  # * 1 Volt
159: calorie = calorie_th = 4.184
160: calorie_IT = 4.1868
161: erg = 1e-7
162: Btu_th = pound * degree_Fahrenheit * calorie_th / gram
163: Btu = Btu_IT = pound * degree_Fahrenheit * calorie_IT / gram
164: ton_TNT = 1e9 * calorie_th
165: # Wh = watt_hour
166: 
167: # power in watt
168: hp = horsepower = 550 * foot * pound * g
169: 
170: # force in newton
171: dyn = dyne = 1e-5
172: lbf = pound_force = pound * g
173: kgf = kilogram_force = g  # * 1 kg
174: 
175: # functions for conversions that are not linear
176: 
177: 
178: def convert_temperature(val, old_scale, new_scale):
179:     '''
180:     Convert from a temperature scale to another one among Celsius, Kelvin,
181:     Fahrenheit and Rankine scales.
182: 
183:     Parameters
184:     ----------
185:     val : array_like
186:         Value(s) of the temperature(s) to be converted expressed in the
187:         original scale.
188: 
189:     old_scale: str
190:         Specifies as a string the original scale from which the temperature
191:         value(s) will be converted. Supported scales are Celsius ('Celsius',
192:         'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
193:         Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine
194:         ('Rankine', 'rankine', 'R', 'r').
195: 
196:     new_scale: str
197:         Specifies as a string the new scale to which the temperature
198:         value(s) will be converted. Supported scales are Celsius ('Celsius',
199:         'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
200:         Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine
201:         ('Rankine', 'rankine', 'R', 'r').
202: 
203:     Returns
204:     -------
205:     res : float or array of floats
206:         Value(s) of the converted temperature(s) expressed in the new scale.
207: 
208:     Notes
209:     -----
210:     .. versionadded:: 0.18.0
211: 
212:     Examples
213:     --------
214:     >>> from scipy.constants import convert_temperature
215:     >>> convert_temperature(np.array([-40, 40.0]), 'Celsius', 'Kelvin')
216:     array([ 233.15,  313.15])
217: 
218:     '''
219:     # Convert from `old_scale` to Kelvin
220:     if old_scale.lower() in ['celsius', 'c']:
221:         tempo = _np.asanyarray(val) + zero_Celsius
222:     elif old_scale.lower() in ['kelvin', 'k']:
223:         tempo = _np.asanyarray(val)
224:     elif old_scale.lower() in ['fahrenheit', 'f']:
225:         tempo = (_np.asanyarray(val) - 32.) * 5. / 9. + zero_Celsius
226:     elif old_scale.lower() in ['rankine', 'r']:
227:         tempo = _np.asanyarray(val) * 5. / 9.
228:     else:
229:         raise NotImplementedError("%s scale is unsupported: supported scales "
230:                                   "are Celsius, Kelvin, Fahrenheit and "
231:                                   "Rankine" % old_scale)
232:     # and from Kelvin to `new_scale`.
233:     if new_scale.lower() in ['celsius', 'c']:
234:         res = tempo - zero_Celsius
235:     elif new_scale.lower() in ['kelvin', 'k']:
236:         res = tempo
237:     elif new_scale.lower() in ['fahrenheit', 'f']:
238:         res = (tempo - zero_Celsius) * 9. / 5. + 32.
239:     elif new_scale.lower() in ['rankine', 'r']:
240:         res = tempo * 9. / 5.
241:     else:
242:         raise NotImplementedError("'%s' scale is unsupported: supported "
243:                                   "scales are 'Celsius', 'Kelvin', "
244:                                   "'Fahrenheit' and 'Rankine'" % new_scale)
245: 
246:     return res
247: 
248: 
249: # optics
250: 
251: 
252: def lambda2nu(lambda_):
253:     '''
254:     Convert wavelength to optical frequency
255: 
256:     Parameters
257:     ----------
258:     lambda_ : array_like
259:         Wavelength(s) to be converted.
260: 
261:     Returns
262:     -------
263:     nu : float or array of floats
264:         Equivalent optical frequency.
265: 
266:     Notes
267:     -----
268:     Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
269:     (vacuum) speed of light in meters/second.
270: 
271:     Examples
272:     --------
273:     >>> from scipy.constants import lambda2nu, speed_of_light
274:     >>> lambda2nu(np.array((1, speed_of_light)))
275:     array([  2.99792458e+08,   1.00000000e+00])
276: 
277:     '''
278:     return _np.asanyarray(c) / lambda_
279: 
280: 
281: def nu2lambda(nu):
282:     '''
283:     Convert optical frequency to wavelength.
284: 
285:     Parameters
286:     ----------
287:     nu : array_like
288:         Optical frequency to be converted.
289: 
290:     Returns
291:     -------
292:     lambda : float or array of floats
293:         Equivalent wavelength(s).
294: 
295:     Notes
296:     -----
297:     Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the
298:     (vacuum) speed of light in meters/second.
299: 
300:     Examples
301:     --------
302:     >>> from scipy.constants import nu2lambda, speed_of_light
303:     >>> nu2lambda(np.array((1, speed_of_light)))
304:     array([  2.99792458e+08,   1.00000000e+00])
305: 
306:     '''
307:     return c / _np.asanyarray(nu)
308: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_13861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', "\nCollection of physical constants and conversion factors.\n\nMost constants are in SI units, so you can do\nprint '10 mile per minute is', 10*mile/minute, 'm/s or', 10*mile/(minute*knot), 'knots'\n\nThe list is not meant to be comprehensive, but just a convenient list for everyday use.\n")
str_13862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\nBasSw 2006\nphysical constants: imported from CODATA\nunit conversion: see e.g. NIST special publication 811\nUse at own risk: double-check values before calculating your Mars orbit-insertion burn.\nSome constants exist in a few variants, which are marked with suffixes.\nThe ones without any suffix should be the most common one.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import math' statement (line 20)
import math as _math

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), '_math', _math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.constants.codata import _cd' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/')
import_13863 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.constants.codata')

if (type(import_13863) is not StypyTypeError):

    if (import_13863 != 'pyd_module'):
        __import__(import_13863)
        sys_modules_13864 = sys.modules[import_13863]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.constants.codata', sys_modules_13864.module_type_store, module_type_store, ['value'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_13864, sys_modules_13864.module_type_store, module_type_store)
    else:
        from scipy.constants.codata import value as _cd

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.constants.codata', None, module_type_store, ['value'], [_cd])

else:
    # Assigning a type to the variable 'scipy.constants.codata' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.constants.codata', import_13863)

# Adding an alias
module_type_store.add_alias('_cd', 'value')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import numpy' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/')
import_13865 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy')

if (type(import_13865) is not StypyTypeError):

    if (import_13865 != 'pyd_module'):
        __import__(import_13865)
        sys_modules_13866 = sys.modules[import_13865]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), '_np', sys_modules_13866.module_type_store, module_type_store)
    else:
        import numpy as _np

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), '_np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', import_13865)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/')


# Assigning a Attribute to a Name (line 25):
# Getting the type of '_math' (line 25)
_math_13867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), '_math')
# Obtaining the member 'pi' of a type (line 25)
pi_13868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), _math_13867, 'pi')
# Assigning a type to the variable 'pi' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'pi', pi_13868)

# Multiple assignment of 2 elements.
int_13869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')

# Call to sqrt(...): (line 26)
# Processing the call arguments (line 26)
int_13872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'int')
# Processing the call keyword arguments (line 26)
kwargs_13873 = {}
# Getting the type of '_math' (line 26)
_math_13870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), '_math', False)
# Obtaining the member 'sqrt' of a type (line 26)
sqrt_13871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 29), _math_13870, 'sqrt')
# Calling sqrt(args, kwargs) (line 26)
sqrt_call_result_13874 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), sqrt_13871, *[int_13872], **kwargs_13873)

# Applying the binary operator '+' (line 26)
result_add_13875 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 25), '+', int_13869, sqrt_call_result_13874)

int_13876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 46), 'int')
# Applying the binary operator 'div' (line 26)
result_div_13877 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 24), 'div', result_add_13875, int_13876)

# Assigning a type to the variable 'golden_ratio' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'golden_ratio', result_div_13877)
# Getting the type of 'golden_ratio' (line 26)
golden_ratio_13878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'golden_ratio')
# Assigning a type to the variable 'golden' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'golden', golden_ratio_13878)

# Assigning a Num to a Name (line 29):
float_13879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'float')
# Assigning a type to the variable 'yotta' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'yotta', float_13879)

# Assigning a Num to a Name (line 30):
float_13880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'float')
# Assigning a type to the variable 'zetta' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'zetta', float_13880)

# Assigning a Num to a Name (line 31):
float_13881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 6), 'float')
# Assigning a type to the variable 'exa' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'exa', float_13881)

# Assigning a Num to a Name (line 32):
float_13882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 7), 'float')
# Assigning a type to the variable 'peta' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'peta', float_13882)

# Assigning a Num to a Name (line 33):
float_13883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 7), 'float')
# Assigning a type to the variable 'tera' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'tera', float_13883)

# Assigning a Num to a Name (line 34):
float_13884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 7), 'float')
# Assigning a type to the variable 'giga' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'giga', float_13884)

# Assigning a Num to a Name (line 35):
float_13885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 7), 'float')
# Assigning a type to the variable 'mega' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'mega', float_13885)

# Assigning a Num to a Name (line 36):
float_13886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 7), 'float')
# Assigning a type to the variable 'kilo' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'kilo', float_13886)

# Assigning a Num to a Name (line 37):
float_13887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'float')
# Assigning a type to the variable 'hecto' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'hecto', float_13887)

# Assigning a Num to a Name (line 38):
float_13888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 7), 'float')
# Assigning a type to the variable 'deka' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'deka', float_13888)

# Assigning a Num to a Name (line 39):
float_13889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 7), 'float')
# Assigning a type to the variable 'deci' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'deci', float_13889)

# Assigning a Num to a Name (line 40):
float_13890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'float')
# Assigning a type to the variable 'centi' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'centi', float_13890)

# Assigning a Num to a Name (line 41):
float_13891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'float')
# Assigning a type to the variable 'milli' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'milli', float_13891)

# Assigning a Num to a Name (line 42):
float_13892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'float')
# Assigning a type to the variable 'micro' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'micro', float_13892)

# Assigning a Num to a Name (line 43):
float_13893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 7), 'float')
# Assigning a type to the variable 'nano' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'nano', float_13893)

# Assigning a Num to a Name (line 44):
float_13894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 7), 'float')
# Assigning a type to the variable 'pico' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'pico', float_13894)

# Assigning a Num to a Name (line 45):
float_13895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'float')
# Assigning a type to the variable 'femto' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'femto', float_13895)

# Assigning a Num to a Name (line 46):
float_13896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 7), 'float')
# Assigning a type to the variable 'atto' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'atto', float_13896)

# Assigning a Num to a Name (line 47):
float_13897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'float')
# Assigning a type to the variable 'zepto' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'zepto', float_13897)

# Assigning a BinOp to a Name (line 50):
int_13898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 7), 'int')
int_13899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'int')
# Applying the binary operator '**' (line 50)
result_pow_13900 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), '**', int_13898, int_13899)

# Assigning a type to the variable 'kibi' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'kibi', result_pow_13900)

# Assigning a BinOp to a Name (line 51):
int_13901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 7), 'int')
int_13902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 10), 'int')
# Applying the binary operator '**' (line 51)
result_pow_13903 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '**', int_13901, int_13902)

# Assigning a type to the variable 'mebi' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'mebi', result_pow_13903)

# Assigning a BinOp to a Name (line 52):
int_13904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 7), 'int')
int_13905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 10), 'int')
# Applying the binary operator '**' (line 52)
result_pow_13906 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), '**', int_13904, int_13905)

# Assigning a type to the variable 'gibi' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'gibi', result_pow_13906)

# Assigning a BinOp to a Name (line 53):
int_13907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 7), 'int')
int_13908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 10), 'int')
# Applying the binary operator '**' (line 53)
result_pow_13909 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), '**', int_13907, int_13908)

# Assigning a type to the variable 'tebi' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'tebi', result_pow_13909)

# Assigning a BinOp to a Name (line 54):
int_13910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 7), 'int')
int_13911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 10), 'int')
# Applying the binary operator '**' (line 54)
result_pow_13912 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), '**', int_13910, int_13911)

# Assigning a type to the variable 'pebi' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'pebi', result_pow_13912)

# Assigning a BinOp to a Name (line 55):
int_13913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 7), 'int')
int_13914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 10), 'int')
# Applying the binary operator '**' (line 55)
result_pow_13915 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), '**', int_13913, int_13914)

# Assigning a type to the variable 'exbi' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'exbi', result_pow_13915)

# Assigning a BinOp to a Name (line 56):
int_13916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 7), 'int')
int_13917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 10), 'int')
# Applying the binary operator '**' (line 56)
result_pow_13918 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 7), '**', int_13916, int_13917)

# Assigning a type to the variable 'zebi' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'zebi', result_pow_13918)

# Assigning a BinOp to a Name (line 57):
int_13919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 7), 'int')
int_13920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'int')
# Applying the binary operator '**' (line 57)
result_pow_13921 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), '**', int_13919, int_13920)

# Assigning a type to the variable 'yobi' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'yobi', result_pow_13921)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 60)
# Processing the call arguments (line 60)
str_13923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'str', 'speed of light in vacuum')
# Processing the call keyword arguments (line 60)
kwargs_13924 = {}
# Getting the type of '_cd' (line 60)
_cd_13922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), '_cd', False)
# Calling _cd(args, kwargs) (line 60)
_cd_call_result_13925 = invoke(stypy.reporting.localization.Localization(__file__, 60, 21), _cd_13922, *[str_13923], **kwargs_13924)

# Assigning a type to the variable 'speed_of_light' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'speed_of_light', _cd_call_result_13925)
# Getting the type of 'speed_of_light' (line 60)
speed_of_light_13926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'speed_of_light')
# Assigning a type to the variable 'c' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'c', speed_of_light_13926)

# Assigning a BinOp to a Name (line 61):
float_13927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 7), 'float')
# Getting the type of 'pi' (line 61)
pi_13928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'pi')
# Applying the binary operator '*' (line 61)
result_mul_13929 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), '*', float_13927, pi_13928)

# Assigning a type to the variable 'mu_0' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'mu_0', result_mul_13929)

# Assigning a BinOp to a Name (line 62):
int_13930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'int')
# Getting the type of 'mu_0' (line 62)
mu_0_13931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'mu_0')
# Getting the type of 'c' (line 62)
c_13932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'c')
# Applying the binary operator '*' (line 62)
result_mul_13933 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 17), '*', mu_0_13931, c_13932)

# Getting the type of 'c' (line 62)
c_13934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'c')
# Applying the binary operator '*' (line 62)
result_mul_13935 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 23), '*', result_mul_13933, c_13934)

# Applying the binary operator 'div' (line 62)
result_div_13936 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), 'div', int_13930, result_mul_13935)

# Assigning a type to the variable 'epsilon_0' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'epsilon_0', result_div_13936)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 63)
# Processing the call arguments (line 63)
str_13938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 17), 'str', 'Planck constant')
# Processing the call keyword arguments (line 63)
kwargs_13939 = {}
# Getting the type of '_cd' (line 63)
_cd_13937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), '_cd', False)
# Calling _cd(args, kwargs) (line 63)
_cd_call_result_13940 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), _cd_13937, *[str_13938], **kwargs_13939)

# Assigning a type to the variable 'Planck' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'Planck', _cd_call_result_13940)
# Getting the type of 'Planck' (line 63)
Planck_13941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'Planck')
# Assigning a type to the variable 'h' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'h', Planck_13941)

# Assigning a BinOp to a Name (line 64):
# Getting the type of 'h' (line 64)
h_13942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 7), 'h')
int_13943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'int')
# Getting the type of 'pi' (line 64)
pi_13944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'pi')
# Applying the binary operator '*' (line 64)
result_mul_13945 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '*', int_13943, pi_13944)

# Applying the binary operator 'div' (line 64)
result_div_13946 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 7), 'div', h_13942, result_mul_13945)

# Assigning a type to the variable 'hbar' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'hbar', result_div_13946)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 65)
# Processing the call arguments (line 65)
str_13948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'str', 'Newtonian constant of gravitation')
# Processing the call keyword arguments (line 65)
kwargs_13949 = {}
# Getting the type of '_cd' (line 65)
_cd_13947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), '_cd', False)
# Calling _cd(args, kwargs) (line 65)
_cd_call_result_13950 = invoke(stypy.reporting.localization.Localization(__file__, 65, 29), _cd_13947, *[str_13948], **kwargs_13949)

# Assigning a type to the variable 'gravitational_constant' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'gravitational_constant', _cd_call_result_13950)
# Getting the type of 'gravitational_constant' (line 65)
gravitational_constant_13951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'gravitational_constant')
# Assigning a type to the variable 'G' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'G', gravitational_constant_13951)

# Assigning a Call to a Name (line 66):

# Call to _cd(...): (line 66)
# Processing the call arguments (line 66)
str_13953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'str', 'standard acceleration of gravity')
# Processing the call keyword arguments (line 66)
kwargs_13954 = {}
# Getting the type of '_cd' (line 66)
_cd_13952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), '_cd', False)
# Calling _cd(args, kwargs) (line 66)
_cd_call_result_13955 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), _cd_13952, *[str_13953], **kwargs_13954)

# Assigning a type to the variable 'g' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'g', _cd_call_result_13955)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 67)
# Processing the call arguments (line 67)
str_13957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'str', 'elementary charge')
# Processing the call keyword arguments (line 67)
kwargs_13958 = {}
# Getting the type of '_cd' (line 67)
_cd_13956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), '_cd', False)
# Calling _cd(args, kwargs) (line 67)
_cd_call_result_13959 = invoke(stypy.reporting.localization.Localization(__file__, 67, 24), _cd_13956, *[str_13957], **kwargs_13958)

# Assigning a type to the variable 'elementary_charge' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'elementary_charge', _cd_call_result_13959)
# Getting the type of 'elementary_charge' (line 67)
elementary_charge_13960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'elementary_charge')
# Assigning a type to the variable 'e' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'e', elementary_charge_13960)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 68)
# Processing the call arguments (line 68)
str_13962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'str', 'molar gas constant')
# Processing the call keyword arguments (line 68)
kwargs_13963 = {}
# Getting the type of '_cd' (line 68)
_cd_13961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), '_cd', False)
# Calling _cd(args, kwargs) (line 68)
_cd_call_result_13964 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), _cd_13961, *[str_13962], **kwargs_13963)

# Assigning a type to the variable 'gas_constant' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'gas_constant', _cd_call_result_13964)
# Getting the type of 'gas_constant' (line 68)
gas_constant_13965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'gas_constant')
# Assigning a type to the variable 'R' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'R', gas_constant_13965)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 69)
# Processing the call arguments (line 69)
str_13967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'str', 'fine-structure constant')
# Processing the call keyword arguments (line 69)
kwargs_13968 = {}
# Getting the type of '_cd' (line 69)
_cd_13966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), '_cd', False)
# Calling _cd(args, kwargs) (line 69)
_cd_call_result_13969 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), _cd_13966, *[str_13967], **kwargs_13968)

# Assigning a type to the variable 'fine_structure' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'fine_structure', _cd_call_result_13969)
# Getting the type of 'fine_structure' (line 69)
fine_structure_13970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'fine_structure')
# Assigning a type to the variable 'alpha' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'alpha', fine_structure_13970)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 70)
# Processing the call arguments (line 70)
str_13972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'str', 'Avogadro constant')
# Processing the call keyword arguments (line 70)
kwargs_13973 = {}
# Getting the type of '_cd' (line 70)
_cd_13971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), '_cd', False)
# Calling _cd(args, kwargs) (line 70)
_cd_call_result_13974 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), _cd_13971, *[str_13972], **kwargs_13973)

# Assigning a type to the variable 'Avogadro' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 6), 'Avogadro', _cd_call_result_13974)
# Getting the type of 'Avogadro' (line 70)
Avogadro_13975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 6), 'Avogadro')
# Assigning a type to the variable 'N_A' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'N_A', Avogadro_13975)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 71)
# Processing the call arguments (line 71)
str_13977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'str', 'Boltzmann constant')
# Processing the call keyword arguments (line 71)
kwargs_13978 = {}
# Getting the type of '_cd' (line 71)
_cd_13976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), '_cd', False)
# Calling _cd(args, kwargs) (line 71)
_cd_call_result_13979 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), _cd_13976, *[str_13977], **kwargs_13978)

# Assigning a type to the variable 'Boltzmann' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'Boltzmann', _cd_call_result_13979)
# Getting the type of 'Boltzmann' (line 71)
Boltzmann_13980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'Boltzmann')
# Assigning a type to the variable 'k' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'k', Boltzmann_13980)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 72)
# Processing the call arguments (line 72)
str_13982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'str', 'Stefan-Boltzmann constant')
# Processing the call keyword arguments (line 72)
kwargs_13983 = {}
# Getting the type of '_cd' (line 72)
_cd_13981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), '_cd', False)
# Calling _cd(args, kwargs) (line 72)
_cd_call_result_13984 = invoke(stypy.reporting.localization.Localization(__file__, 72, 27), _cd_13981, *[str_13982], **kwargs_13983)

# Assigning a type to the variable 'Stefan_Boltzmann' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'Stefan_Boltzmann', _cd_call_result_13984)
# Getting the type of 'Stefan_Boltzmann' (line 72)
Stefan_Boltzmann_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'Stefan_Boltzmann')
# Assigning a type to the variable 'sigma' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'sigma', Stefan_Boltzmann_13985)

# Assigning a Call to a Name (line 73):

# Call to _cd(...): (line 73)
# Processing the call arguments (line 73)
str_13987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 11), 'str', 'Wien wavelength displacement law constant')
# Processing the call keyword arguments (line 73)
kwargs_13988 = {}
# Getting the type of '_cd' (line 73)
_cd_13986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), '_cd', False)
# Calling _cd(args, kwargs) (line 73)
_cd_call_result_13989 = invoke(stypy.reporting.localization.Localization(__file__, 73, 7), _cd_13986, *[str_13987], **kwargs_13988)

# Assigning a type to the variable 'Wien' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'Wien', _cd_call_result_13989)

# Assigning a Call to a Name (line 74):

# Call to _cd(...): (line 74)
# Processing the call arguments (line 74)
str_13991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'str', 'Rydberg constant')
# Processing the call keyword arguments (line 74)
kwargs_13992 = {}
# Getting the type of '_cd' (line 74)
_cd_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 10), '_cd', False)
# Calling _cd(args, kwargs) (line 74)
_cd_call_result_13993 = invoke(stypy.reporting.localization.Localization(__file__, 74, 10), _cd_13990, *[str_13991], **kwargs_13992)

# Assigning a type to the variable 'Rydberg' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'Rydberg', _cd_call_result_13993)

# Assigning a Num to a Name (line 77):
float_13994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 7), 'float')
# Assigning a type to the variable 'gram' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'gram', float_13994)

# Assigning a Num to a Name (line 78):
float_13995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'float')
# Assigning a type to the variable 'metric_ton' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'metric_ton', float_13995)

# Assigning a Num to a Name (line 79):
float_13996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'float')
# Assigning a type to the variable 'grain' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'grain', float_13996)

# Multiple assignment of 2 elements.
int_13997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'int')
# Getting the type of 'grain' (line 80)
grain_13998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'grain')
# Applying the binary operator '*' (line 80)
result_mul_13999 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '*', int_13997, grain_13998)

# Assigning a type to the variable 'pound' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 5), 'pound', result_mul_13999)
# Getting the type of 'pound' (line 80)
pound_14000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 5), 'pound')
# Assigning a type to the variable 'lb' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'lb', pound_14000)

# Multiple assignment of 2 elements.
# Getting the type of 'pound' (line 81)
pound_14001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'pound')
# Getting the type of 'g' (line 81)
g_14002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'g')
# Applying the binary operator '*' (line 81)
result_mul_14003 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 16), '*', pound_14001, g_14002)

float_14004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'float')
# Applying the binary operator 'div' (line 81)
result_div_14005 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 26), 'div', result_mul_14003, float_14004)

# Assigning a type to the variable 'slinch' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'slinch', result_div_14005)
# Getting the type of 'slinch' (line 81)
slinch_14006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'slinch')
# Assigning a type to the variable 'blob' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'blob', slinch_14006)

# Assigning a BinOp to a Name (line 82):
# Getting the type of 'blob' (line 82)
blob_14007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'blob')
int_14008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 14), 'int')
# Applying the binary operator 'div' (line 82)
result_div_14009 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 7), 'div', blob_14007, int_14008)

# Assigning a type to the variable 'slug' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'slug', result_div_14009)

# Multiple assignment of 2 elements.
# Getting the type of 'pound' (line 83)
pound_14010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'pound')
int_14011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'int')
# Applying the binary operator 'div' (line 83)
result_div_14012 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 13), 'div', pound_14010, int_14011)

# Assigning a type to the variable 'ounce' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 5), 'ounce', result_div_14012)
# Getting the type of 'ounce' (line 83)
ounce_14013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 5), 'ounce')
# Assigning a type to the variable 'oz' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'oz', ounce_14013)

# Assigning a BinOp to a Name (line 84):
int_14014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
# Getting the type of 'pound' (line 84)
pound_14015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'pound')
# Applying the binary operator '*' (line 84)
result_mul_14016 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 8), '*', int_14014, pound_14015)

# Assigning a type to the variable 'stone' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stone', result_mul_14016)

# Assigning a BinOp to a Name (line 85):
int_14017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'int')
# Getting the type of 'pound' (line 85)
pound_14018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'pound')
# Applying the binary operator '*' (line 85)
result_mul_14019 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), '*', int_14017, pound_14018)

# Assigning a type to the variable 'long_ton' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'long_ton', result_mul_14019)

# Assigning a BinOp to a Name (line 86):
int_14020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'int')
# Getting the type of 'pound' (line 86)
pound_14021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'pound')
# Applying the binary operator '*' (line 86)
result_mul_14022 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '*', int_14020, pound_14021)

# Assigning a type to the variable 'short_ton' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'short_ton', result_mul_14022)

# Assigning a BinOp to a Name (line 88):
int_14023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'int')
# Getting the type of 'grain' (line 88)
grain_14024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'grain')
# Applying the binary operator '*' (line 88)
result_mul_14025 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '*', int_14023, grain_14024)

# Assigning a type to the variable 'troy_ounce' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'troy_ounce', result_mul_14025)

# Assigning a BinOp to a Name (line 89):
int_14026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'int')
# Getting the type of 'troy_ounce' (line 89)
troy_ounce_14027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'troy_ounce')
# Applying the binary operator '*' (line 89)
result_mul_14028 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '*', int_14026, troy_ounce_14027)

# Assigning a type to the variable 'troy_pound' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'troy_pound', result_mul_14028)

# Assigning a Num to a Name (line 90):
float_14029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'float')
# Assigning a type to the variable 'carat' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'carat', float_14029)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 92)
# Processing the call arguments (line 92)
str_14031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'str', 'electron mass')
# Processing the call keyword arguments (line 92)
kwargs_14032 = {}
# Getting the type of '_cd' (line 92)
_cd_14030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), '_cd', False)
# Calling _cd(args, kwargs) (line 92)
_cd_call_result_14033 = invoke(stypy.reporting.localization.Localization(__file__, 92, 22), _cd_14030, *[str_14031], **kwargs_14032)

# Assigning a type to the variable 'electron_mass' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 6), 'electron_mass', _cd_call_result_14033)
# Getting the type of 'electron_mass' (line 92)
electron_mass_14034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 6), 'electron_mass')
# Assigning a type to the variable 'm_e' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'm_e', electron_mass_14034)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 93)
# Processing the call arguments (line 93)
str_14036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'str', 'proton mass')
# Processing the call keyword arguments (line 93)
kwargs_14037 = {}
# Getting the type of '_cd' (line 93)
_cd_14035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), '_cd', False)
# Calling _cd(args, kwargs) (line 93)
_cd_call_result_14038 = invoke(stypy.reporting.localization.Localization(__file__, 93, 20), _cd_14035, *[str_14036], **kwargs_14037)

# Assigning a type to the variable 'proton_mass' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 6), 'proton_mass', _cd_call_result_14038)
# Getting the type of 'proton_mass' (line 93)
proton_mass_14039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 6), 'proton_mass')
# Assigning a type to the variable 'm_p' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'm_p', proton_mass_14039)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 94)
# Processing the call arguments (line 94)
str_14041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'neutron mass')
# Processing the call keyword arguments (line 94)
kwargs_14042 = {}
# Getting the type of '_cd' (line 94)
_cd_14040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), '_cd', False)
# Calling _cd(args, kwargs) (line 94)
_cd_call_result_14043 = invoke(stypy.reporting.localization.Localization(__file__, 94, 21), _cd_14040, *[str_14041], **kwargs_14042)

# Assigning a type to the variable 'neutron_mass' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'neutron_mass', _cd_call_result_14043)
# Getting the type of 'neutron_mass' (line 94)
neutron_mass_14044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'neutron_mass')
# Assigning a type to the variable 'm_n' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'm_n', neutron_mass_14044)

# Multiple assignment of 3 elements.

# Call to _cd(...): (line 95)
# Processing the call arguments (line 95)
str_14046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'str', 'atomic mass constant')
# Processing the call keyword arguments (line 95)
kwargs_14047 = {}
# Getting the type of '_cd' (line 95)
_cd_14045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), '_cd', False)
# Calling _cd(args, kwargs) (line 95)
_cd_call_result_14048 = invoke(stypy.reporting.localization.Localization(__file__, 95, 24), _cd_14045, *[str_14046], **kwargs_14047)

# Assigning a type to the variable 'atomic_mass' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'atomic_mass', _cd_call_result_14048)
# Getting the type of 'atomic_mass' (line 95)
atomic_mass_14049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'atomic_mass')
# Assigning a type to the variable 'u' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 6), 'u', atomic_mass_14049)
# Getting the type of 'u' (line 95)
u_14050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 6), 'u')
# Assigning a type to the variable 'm_u' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'm_u', u_14050)

# Assigning a BinOp to a Name (line 98):
# Getting the type of 'pi' (line 98)
pi_14051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'pi')
int_14052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'int')
# Applying the binary operator 'div' (line 98)
result_div_14053 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 9), 'div', pi_14051, int_14052)

# Assigning a type to the variable 'degree' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'degree', result_div_14053)

# Multiple assignment of 2 elements.
# Getting the type of 'degree' (line 99)
degree_14054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'degree')
int_14055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'int')
# Applying the binary operator 'div' (line 99)
result_div_14056 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 21), 'div', degree_14054, int_14055)

# Assigning a type to the variable 'arcminute' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'arcminute', result_div_14056)
# Getting the type of 'arcminute' (line 99)
arcminute_14057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 9), 'arcminute')
# Assigning a type to the variable 'arcmin' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'arcmin', arcminute_14057)

# Multiple assignment of 2 elements.
# Getting the type of 'arcmin' (line 100)
arcmin_14058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'arcmin')
int_14059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'int')
# Applying the binary operator 'div' (line 100)
result_div_14060 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 21), 'div', arcmin_14058, int_14059)

# Assigning a type to the variable 'arcsecond' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'arcsecond', result_div_14060)
# Getting the type of 'arcsecond' (line 100)
arcsecond_14061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'arcsecond')
# Assigning a type to the variable 'arcsec' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'arcsec', arcsecond_14061)

# Assigning a Num to a Name (line 103):
float_14062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'float')
# Assigning a type to the variable 'minute' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'minute', float_14062)

# Assigning a BinOp to a Name (line 104):
int_14063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 7), 'int')
# Getting the type of 'minute' (line 104)
minute_14064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'minute')
# Applying the binary operator '*' (line 104)
result_mul_14065 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), '*', int_14063, minute_14064)

# Assigning a type to the variable 'hour' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'hour', result_mul_14065)

# Assigning a BinOp to a Name (line 105):
int_14066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 6), 'int')
# Getting the type of 'hour' (line 105)
hour_14067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'hour')
# Applying the binary operator '*' (line 105)
result_mul_14068 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 6), '*', int_14066, hour_14067)

# Assigning a type to the variable 'day' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'day', result_mul_14068)

# Assigning a BinOp to a Name (line 106):
int_14069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 7), 'int')
# Getting the type of 'day' (line 106)
day_14070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'day')
# Applying the binary operator '*' (line 106)
result_mul_14071 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 7), '*', int_14069, day_14070)

# Assigning a type to the variable 'week' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'week', result_mul_14071)

# Assigning a BinOp to a Name (line 107):
int_14072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 7), 'int')
# Getting the type of 'day' (line 107)
day_14073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'day')
# Applying the binary operator '*' (line 107)
result_mul_14074 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 7), '*', int_14072, day_14073)

# Assigning a type to the variable 'year' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'year', result_mul_14074)

# Assigning a BinOp to a Name (line 108):
float_14075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'float')
# Getting the type of 'day' (line 108)
day_14076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'day')
# Applying the binary operator '*' (line 108)
result_mul_14077 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), '*', float_14075, day_14076)

# Assigning a type to the variable 'Julian_year' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'Julian_year', result_mul_14077)

# Assigning a Num to a Name (line 111):
float_14078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 7), 'float')
# Assigning a type to the variable 'inch' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'inch', float_14078)

# Assigning a BinOp to a Name (line 112):
int_14079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 7), 'int')
# Getting the type of 'inch' (line 112)
inch_14080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'inch')
# Applying the binary operator '*' (line 112)
result_mul_14081 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 7), '*', int_14079, inch_14080)

# Assigning a type to the variable 'foot' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'foot', result_mul_14081)

# Assigning a BinOp to a Name (line 113):
int_14082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 7), 'int')
# Getting the type of 'foot' (line 113)
foot_14083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'foot')
# Applying the binary operator '*' (line 113)
result_mul_14084 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 7), '*', int_14082, foot_14083)

# Assigning a type to the variable 'yard' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'yard', result_mul_14084)

# Assigning a BinOp to a Name (line 114):
int_14085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 7), 'int')
# Getting the type of 'yard' (line 114)
yard_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'yard')
# Applying the binary operator '*' (line 114)
result_mul_14087 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), '*', int_14085, yard_14086)

# Assigning a type to the variable 'mile' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'mile', result_mul_14087)

# Assigning a BinOp to a Name (line 115):
# Getting the type of 'inch' (line 115)
inch_14088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 6), 'inch')
int_14089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 13), 'int')
# Applying the binary operator 'div' (line 115)
result_div_14090 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 6), 'div', inch_14088, int_14089)

# Assigning a type to the variable 'mil' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'mil', result_div_14090)

# Multiple assignment of 2 elements.
# Getting the type of 'inch' (line 116)
inch_14091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'inch')
int_14092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'int')
# Applying the binary operator 'div' (line 116)
result_div_14093 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 13), 'div', inch_14091, int_14092)

# Assigning a type to the variable 'point' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 5), 'point', result_div_14093)
# Getting the type of 'point' (line 116)
point_14094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 5), 'point')
# Assigning a type to the variable 'pt' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'pt', point_14094)

# Assigning a BinOp to a Name (line 117):
float_14095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 14), 'float')
int_14096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'int')
# Applying the binary operator 'div' (line 117)
result_div_14097 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 14), 'div', float_14095, int_14096)

# Assigning a type to the variable 'survey_foot' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'survey_foot', result_div_14097)

# Assigning a BinOp to a Name (line 118):
int_14098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 14), 'int')
# Getting the type of 'survey_foot' (line 118)
survey_foot_14099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'survey_foot')
# Applying the binary operator '*' (line 118)
result_mul_14100 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 14), '*', int_14098, survey_foot_14099)

# Assigning a type to the variable 'survey_mile' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'survey_mile', result_mul_14100)

# Assigning a Num to a Name (line 119):
float_14101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'float')
# Assigning a type to the variable 'nautical_mile' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'nautical_mile', float_14101)

# Assigning a Num to a Name (line 120):
float_14102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'float')
# Assigning a type to the variable 'fermi' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'fermi', float_14102)

# Assigning a Num to a Name (line 121):
float_14103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 11), 'float')
# Assigning a type to the variable 'angstrom' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'angstrom', float_14103)

# Assigning a Num to a Name (line 122):
float_14104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 9), 'float')
# Assigning a type to the variable 'micron' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'micron', float_14104)

# Multiple assignment of 2 elements.
float_14105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'float')
# Assigning a type to the variable 'astronomical_unit' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 5), 'astronomical_unit', float_14105)
# Getting the type of 'astronomical_unit' (line 123)
astronomical_unit_14106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 5), 'astronomical_unit')
# Assigning a type to the variable 'au' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'au', astronomical_unit_14106)

# Assigning a BinOp to a Name (line 124):
# Getting the type of 'Julian_year' (line 124)
Julian_year_14107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'Julian_year')
# Getting the type of 'c' (line 124)
c_14108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'c')
# Applying the binary operator '*' (line 124)
result_mul_14109 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), '*', Julian_year_14107, c_14108)

# Assigning a type to the variable 'light_year' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'light_year', result_mul_14109)

# Assigning a BinOp to a Name (line 125):
# Getting the type of 'au' (line 125)
au_14110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'au')
# Getting the type of 'arcsec' (line 125)
arcsec_14111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'arcsec')
# Applying the binary operator 'div' (line 125)
result_div_14112 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 9), 'div', au_14110, arcsec_14111)

# Assigning a type to the variable 'parsec' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'parsec', result_div_14112)

# Multiple assignment of 2 elements.

# Call to _cd(...): (line 128)
# Processing the call arguments (line 128)
str_14114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'str', 'standard atmosphere')
# Processing the call keyword arguments (line 128)
kwargs_14115 = {}
# Getting the type of '_cd' (line 128)
_cd_14113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), '_cd', False)
# Calling _cd(args, kwargs) (line 128)
_cd_call_result_14116 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), _cd_14113, *[str_14114], **kwargs_14115)

# Assigning a type to the variable 'atmosphere' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 6), 'atmosphere', _cd_call_result_14116)
# Getting the type of 'atmosphere' (line 128)
atmosphere_14117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 6), 'atmosphere')
# Assigning a type to the variable 'atm' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'atm', atmosphere_14117)

# Assigning a Num to a Name (line 129):
float_14118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 6), 'float')
# Assigning a type to the variable 'bar' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'bar', float_14118)

# Multiple assignment of 2 elements.
# Getting the type of 'atm' (line 130)
atm_14119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 14), 'atm')
int_14120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 20), 'int')
# Applying the binary operator 'div' (line 130)
result_div_14121 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 14), 'div', atm_14119, int_14120)

# Assigning a type to the variable 'mmHg' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'mmHg', result_div_14121)
# Getting the type of 'mmHg' (line 130)
mmHg_14122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'mmHg')
# Assigning a type to the variable 'torr' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'torr', mmHg_14122)

# Assigning a BinOp to a Name (line 131):
# Getting the type of 'pound' (line 131)
pound_14123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 6), 'pound')
# Getting the type of 'g' (line 131)
g_14124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'g')
# Applying the binary operator '*' (line 131)
result_mul_14125 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 6), '*', pound_14123, g_14124)

# Getting the type of 'inch' (line 131)
inch_14126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'inch')
# Getting the type of 'inch' (line 131)
inch_14127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'inch')
# Applying the binary operator '*' (line 131)
result_mul_14128 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 19), '*', inch_14126, inch_14127)

# Applying the binary operator 'div' (line 131)
result_div_14129 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), 'div', result_mul_14125, result_mul_14128)

# Assigning a type to the variable 'psi' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'psi', result_div_14129)

# Assigning a Num to a Name (line 134):
float_14130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 10), 'float')
# Assigning a type to the variable 'hectare' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'hectare', float_14130)

# Assigning a BinOp to a Name (line 135):
int_14131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 7), 'int')
# Getting the type of 'foot' (line 135)
foot_14132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'foot')
int_14133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'int')
# Applying the binary operator '**' (line 135)
result_pow_14134 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '**', foot_14132, int_14133)

# Applying the binary operator '*' (line 135)
result_mul_14135 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 7), '*', int_14131, result_pow_14134)

# Assigning a type to the variable 'acre' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'acre', result_mul_14135)

# Multiple assignment of 2 elements.
float_14136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'float')
# Assigning a type to the variable 'liter' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'liter', float_14136)
# Getting the type of 'liter' (line 138)
liter_14137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'liter')
# Assigning a type to the variable 'litre' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'litre', liter_14137)

# Multiple assignment of 2 elements.
int_14138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'int')
# Getting the type of 'inch' (line 139)
inch_14139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'inch')
int_14140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 33), 'int')
# Applying the binary operator '**' (line 139)
result_pow_14141 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 27), '**', inch_14139, int_14140)

# Applying the binary operator '*' (line 139)
result_mul_14142 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '*', int_14138, result_pow_14141)

# Assigning a type to the variable 'gallon_US' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'gallon_US', result_mul_14142)
# Getting the type of 'gallon_US' (line 139)
gallon_US_14143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'gallon_US')
# Assigning a type to the variable 'gallon' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'gallon', gallon_US_14143)

# Multiple assignment of 2 elements.
# Getting the type of 'gallon_US' (line 141)
gallon_US_14144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'gallon_US')
int_14145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 43), 'int')
# Applying the binary operator 'div' (line 141)
result_div_14146 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 31), 'div', gallon_US_14144, int_14145)

# Assigning a type to the variable 'fluid_ounce_US' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 14), 'fluid_ounce_US', result_div_14146)
# Getting the type of 'fluid_ounce_US' (line 141)
fluid_ounce_US_14147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 14), 'fluid_ounce_US')
# Assigning a type to the variable 'fluid_ounce' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'fluid_ounce', fluid_ounce_US_14147)

# Multiple assignment of 2 elements.
int_14148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'int')
# Getting the type of 'gallon_US' (line 142)
gallon_US_14149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'gallon_US')
# Applying the binary operator '*' (line 142)
result_mul_14150 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 15), '*', int_14148, gallon_US_14149)

# Assigning a type to the variable 'barrel' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 6), 'barrel', result_mul_14150)
# Getting the type of 'barrel' (line 142)
barrel_14151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 6), 'barrel')
# Assigning a type to the variable 'bbl' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'bbl', barrel_14151)

# Assigning a Num to a Name (line 144):
float_14152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 13), 'float')
# Assigning a type to the variable 'gallon_imp' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'gallon_imp', float_14152)

# Assigning a BinOp to a Name (line 145):
# Getting the type of 'gallon_imp' (line 145)
gallon_imp_14153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'gallon_imp')
int_14154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 31), 'int')
# Applying the binary operator 'div' (line 145)
result_div_14155 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 18), 'div', gallon_imp_14153, int_14154)

# Assigning a type to the variable 'fluid_ounce_imp' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'fluid_ounce_imp', result_div_14155)

# Assigning a BinOp to a Name (line 148):
float_14156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 6), 'float')
# Getting the type of 'hour' (line 148)
hour_14157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'hour')
# Applying the binary operator 'div' (line 148)
result_div_14158 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 6), 'div', float_14156, hour_14157)

# Assigning a type to the variable 'kmh' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'kmh', result_div_14158)

# Assigning a BinOp to a Name (line 149):
# Getting the type of 'mile' (line 149)
mile_14159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 6), 'mile')
# Getting the type of 'hour' (line 149)
hour_14160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 13), 'hour')
# Applying the binary operator 'div' (line 149)
result_div_14161 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 6), 'div', mile_14159, hour_14160)

# Assigning a type to the variable 'mph' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'mph', result_div_14161)

# Multiple assignment of 2 elements.
float_14162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 24), 'float')
# Assigning a type to the variable 'speed_of_sound' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'speed_of_sound', float_14162)
# Getting the type of 'speed_of_sound' (line 150)
speed_of_sound_14163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'speed_of_sound')
# Assigning a type to the variable 'mach' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'mach', speed_of_sound_14163)

# Assigning a BinOp to a Name (line 151):
# Getting the type of 'nautical_mile' (line 151)
nautical_mile_14164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'nautical_mile')
# Getting the type of 'hour' (line 151)
hour_14165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'hour')
# Applying the binary operator 'div' (line 151)
result_div_14166 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), 'div', nautical_mile_14164, hour_14165)

# Assigning a type to the variable 'knot' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'knot', result_div_14166)

# Assigning a Num to a Name (line 154):
float_14167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'float')
# Assigning a type to the variable 'zero_Celsius' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'zero_Celsius', float_14167)

# Assigning a BinOp to a Name (line 155):
int_14168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
float_14169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'float')
# Applying the binary operator 'div' (line 155)
result_div_14170 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 20), 'div', int_14168, float_14169)

# Assigning a type to the variable 'degree_Fahrenheit' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'degree_Fahrenheit', result_div_14170)

# Multiple assignment of 2 elements.
# Getting the type of 'elementary_charge' (line 158)
elementary_charge_14171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'elementary_charge')
# Assigning a type to the variable 'electron_volt' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 5), 'electron_volt', elementary_charge_14171)
# Getting the type of 'electron_volt' (line 158)
electron_volt_14172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 5), 'electron_volt')
# Assigning a type to the variable 'eV' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'eV', electron_volt_14172)

# Multiple assignment of 2 elements.
float_14173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 23), 'float')
# Assigning a type to the variable 'calorie_th' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 10), 'calorie_th', float_14173)
# Getting the type of 'calorie_th' (line 159)
calorie_th_14174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 10), 'calorie_th')
# Assigning a type to the variable 'calorie' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'calorie', calorie_th_14174)

# Assigning a Num to a Name (line 160):
float_14175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'float')
# Assigning a type to the variable 'calorie_IT' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'calorie_IT', float_14175)

# Assigning a Num to a Name (line 161):
float_14176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 6), 'float')
# Assigning a type to the variable 'erg' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'erg', float_14176)

# Assigning a BinOp to a Name (line 162):
# Getting the type of 'pound' (line 162)
pound_14177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 9), 'pound')
# Getting the type of 'degree_Fahrenheit' (line 162)
degree_Fahrenheit_14178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'degree_Fahrenheit')
# Applying the binary operator '*' (line 162)
result_mul_14179 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 9), '*', pound_14177, degree_Fahrenheit_14178)

# Getting the type of 'calorie_th' (line 162)
calorie_th_14180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'calorie_th')
# Applying the binary operator '*' (line 162)
result_mul_14181 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 35), '*', result_mul_14179, calorie_th_14180)

# Getting the type of 'gram' (line 162)
gram_14182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 'gram')
# Applying the binary operator 'div' (line 162)
result_div_14183 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 48), 'div', result_mul_14181, gram_14182)

# Assigning a type to the variable 'Btu_th' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'Btu_th', result_div_14183)

# Multiple assignment of 2 elements.
# Getting the type of 'pound' (line 163)
pound_14184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'pound')
# Getting the type of 'degree_Fahrenheit' (line 163)
degree_Fahrenheit_14185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'degree_Fahrenheit')
# Applying the binary operator '*' (line 163)
result_mul_14186 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 15), '*', pound_14184, degree_Fahrenheit_14185)

# Getting the type of 'calorie_IT' (line 163)
calorie_IT_14187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), 'calorie_IT')
# Applying the binary operator '*' (line 163)
result_mul_14188 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 41), '*', result_mul_14186, calorie_IT_14187)

# Getting the type of 'gram' (line 163)
gram_14189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 56), 'gram')
# Applying the binary operator 'div' (line 163)
result_div_14190 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 54), 'div', result_mul_14188, gram_14189)

# Assigning a type to the variable 'Btu_IT' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 6), 'Btu_IT', result_div_14190)
# Getting the type of 'Btu_IT' (line 163)
Btu_IT_14191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 6), 'Btu_IT')
# Assigning a type to the variable 'Btu' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'Btu', Btu_IT_14191)

# Assigning a BinOp to a Name (line 164):
float_14192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 10), 'float')
# Getting the type of 'calorie_th' (line 164)
calorie_th_14193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'calorie_th')
# Applying the binary operator '*' (line 164)
result_mul_14194 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 10), '*', float_14192, calorie_th_14193)

# Assigning a type to the variable 'ton_TNT' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'ton_TNT', result_mul_14194)

# Multiple assignment of 2 elements.
int_14195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'int')
# Getting the type of 'foot' (line 168)
foot_14196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'foot')
# Applying the binary operator '*' (line 168)
result_mul_14197 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 18), '*', int_14195, foot_14196)

# Getting the type of 'pound' (line 168)
pound_14198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'pound')
# Applying the binary operator '*' (line 168)
result_mul_14199 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 29), '*', result_mul_14197, pound_14198)

# Getting the type of 'g' (line 168)
g_14200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'g')
# Applying the binary operator '*' (line 168)
result_mul_14201 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 37), '*', result_mul_14199, g_14200)

# Assigning a type to the variable 'horsepower' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 5), 'horsepower', result_mul_14201)
# Getting the type of 'horsepower' (line 168)
horsepower_14202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 5), 'horsepower')
# Assigning a type to the variable 'hp' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'hp', horsepower_14202)

# Multiple assignment of 2 elements.
float_14203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 13), 'float')
# Assigning a type to the variable 'dyne' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 6), 'dyne', float_14203)
# Getting the type of 'dyne' (line 171)
dyne_14204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 6), 'dyne')
# Assigning a type to the variable 'dyn' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'dyn', dyne_14204)

# Multiple assignment of 2 elements.
# Getting the type of 'pound' (line 172)
pound_14205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'pound')
# Getting the type of 'g' (line 172)
g_14206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'g')
# Applying the binary operator '*' (line 172)
result_mul_14207 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 20), '*', pound_14205, g_14206)

# Assigning a type to the variable 'pound_force' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 6), 'pound_force', result_mul_14207)
# Getting the type of 'pound_force' (line 172)
pound_force_14208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 6), 'pound_force')
# Assigning a type to the variable 'lbf' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'lbf', pound_force_14208)

# Multiple assignment of 2 elements.
# Getting the type of 'g' (line 173)
g_14209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'g')
# Assigning a type to the variable 'kilogram_force' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 6), 'kilogram_force', g_14209)
# Getting the type of 'kilogram_force' (line 173)
kilogram_force_14210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 6), 'kilogram_force')
# Assigning a type to the variable 'kgf' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'kgf', kilogram_force_14210)

@norecursion
def convert_temperature(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'convert_temperature'
    module_type_store = module_type_store.open_function_context('convert_temperature', 178, 0, False)
    
    # Passed parameters checking function
    convert_temperature.stypy_localization = localization
    convert_temperature.stypy_type_of_self = None
    convert_temperature.stypy_type_store = module_type_store
    convert_temperature.stypy_function_name = 'convert_temperature'
    convert_temperature.stypy_param_names_list = ['val', 'old_scale', 'new_scale']
    convert_temperature.stypy_varargs_param_name = None
    convert_temperature.stypy_kwargs_param_name = None
    convert_temperature.stypy_call_defaults = defaults
    convert_temperature.stypy_call_varargs = varargs
    convert_temperature.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convert_temperature', ['val', 'old_scale', 'new_scale'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convert_temperature', localization, ['val', 'old_scale', 'new_scale'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convert_temperature(...)' code ##################

    str_14211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, (-1)), 'str', "\n    Convert from a temperature scale to another one among Celsius, Kelvin,\n    Fahrenheit and Rankine scales.\n\n    Parameters\n    ----------\n    val : array_like\n        Value(s) of the temperature(s) to be converted expressed in the\n        original scale.\n\n    old_scale: str\n        Specifies as a string the original scale from which the temperature\n        value(s) will be converted. Supported scales are Celsius ('Celsius',\n        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),\n        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine\n        ('Rankine', 'rankine', 'R', 'r').\n\n    new_scale: str\n        Specifies as a string the new scale to which the temperature\n        value(s) will be converted. Supported scales are Celsius ('Celsius',\n        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),\n        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine\n        ('Rankine', 'rankine', 'R', 'r').\n\n    Returns\n    -------\n    res : float or array of floats\n        Value(s) of the converted temperature(s) expressed in the new scale.\n\n    Notes\n    -----\n    .. versionadded:: 0.18.0\n\n    Examples\n    --------\n    >>> from scipy.constants import convert_temperature\n    >>> convert_temperature(np.array([-40, 40.0]), 'Celsius', 'Kelvin')\n    array([ 233.15,  313.15])\n\n    ")
    
    
    
    # Call to lower(...): (line 220)
    # Processing the call keyword arguments (line 220)
    kwargs_14214 = {}
    # Getting the type of 'old_scale' (line 220)
    old_scale_14212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 7), 'old_scale', False)
    # Obtaining the member 'lower' of a type (line 220)
    lower_14213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 7), old_scale_14212, 'lower')
    # Calling lower(args, kwargs) (line 220)
    lower_call_result_14215 = invoke(stypy.reporting.localization.Localization(__file__, 220, 7), lower_14213, *[], **kwargs_14214)
    
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_14216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    str_14217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 29), 'str', 'celsius')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 28), list_14216, str_14217)
    # Adding element type (line 220)
    str_14218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 40), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 28), list_14216, str_14218)
    
    # Applying the binary operator 'in' (line 220)
    result_contains_14219 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 7), 'in', lower_call_result_14215, list_14216)
    
    # Testing the type of an if condition (line 220)
    if_condition_14220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 4), result_contains_14219)
    # Assigning a type to the variable 'if_condition_14220' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'if_condition_14220', if_condition_14220)
    # SSA begins for if statement (line 220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 221):
    
    # Call to asanyarray(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'val' (line 221)
    val_14223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'val', False)
    # Processing the call keyword arguments (line 221)
    kwargs_14224 = {}
    # Getting the type of '_np' (line 221)
    _np_14221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), '_np', False)
    # Obtaining the member 'asanyarray' of a type (line 221)
    asanyarray_14222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), _np_14221, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 221)
    asanyarray_call_result_14225 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), asanyarray_14222, *[val_14223], **kwargs_14224)
    
    # Getting the type of 'zero_Celsius' (line 221)
    zero_Celsius_14226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 'zero_Celsius')
    # Applying the binary operator '+' (line 221)
    result_add_14227 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 16), '+', asanyarray_call_result_14225, zero_Celsius_14226)
    
    # Assigning a type to the variable 'tempo' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tempo', result_add_14227)
    # SSA branch for the else part of an if statement (line 220)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 222)
    # Processing the call keyword arguments (line 222)
    kwargs_14230 = {}
    # Getting the type of 'old_scale' (line 222)
    old_scale_14228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 9), 'old_scale', False)
    # Obtaining the member 'lower' of a type (line 222)
    lower_14229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 9), old_scale_14228, 'lower')
    # Calling lower(args, kwargs) (line 222)
    lower_call_result_14231 = invoke(stypy.reporting.localization.Localization(__file__, 222, 9), lower_14229, *[], **kwargs_14230)
    
    
    # Obtaining an instance of the builtin type 'list' (line 222)
    list_14232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 222)
    # Adding element type (line 222)
    str_14233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 31), 'str', 'kelvin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 30), list_14232, str_14233)
    # Adding element type (line 222)
    str_14234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 41), 'str', 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 30), list_14232, str_14234)
    
    # Applying the binary operator 'in' (line 222)
    result_contains_14235 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 9), 'in', lower_call_result_14231, list_14232)
    
    # Testing the type of an if condition (line 222)
    if_condition_14236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 9), result_contains_14235)
    # Assigning a type to the variable 'if_condition_14236' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 9), 'if_condition_14236', if_condition_14236)
    # SSA begins for if statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 223):
    
    # Call to asanyarray(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'val' (line 223)
    val_14239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 31), 'val', False)
    # Processing the call keyword arguments (line 223)
    kwargs_14240 = {}
    # Getting the type of '_np' (line 223)
    _np_14237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), '_np', False)
    # Obtaining the member 'asanyarray' of a type (line 223)
    asanyarray_14238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 16), _np_14237, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 223)
    asanyarray_call_result_14241 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), asanyarray_14238, *[val_14239], **kwargs_14240)
    
    # Assigning a type to the variable 'tempo' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tempo', asanyarray_call_result_14241)
    # SSA branch for the else part of an if statement (line 222)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 224)
    # Processing the call keyword arguments (line 224)
    kwargs_14244 = {}
    # Getting the type of 'old_scale' (line 224)
    old_scale_14242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'old_scale', False)
    # Obtaining the member 'lower' of a type (line 224)
    lower_14243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 9), old_scale_14242, 'lower')
    # Calling lower(args, kwargs) (line 224)
    lower_call_result_14245 = invoke(stypy.reporting.localization.Localization(__file__, 224, 9), lower_14243, *[], **kwargs_14244)
    
    
    # Obtaining an instance of the builtin type 'list' (line 224)
    list_14246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 224)
    # Adding element type (line 224)
    str_14247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 31), 'str', 'fahrenheit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 30), list_14246, str_14247)
    # Adding element type (line 224)
    str_14248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 30), list_14246, str_14248)
    
    # Applying the binary operator 'in' (line 224)
    result_contains_14249 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 9), 'in', lower_call_result_14245, list_14246)
    
    # Testing the type of an if condition (line 224)
    if_condition_14250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 9), result_contains_14249)
    # Assigning a type to the variable 'if_condition_14250' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'if_condition_14250', if_condition_14250)
    # SSA begins for if statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 225):
    
    # Call to asanyarray(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'val' (line 225)
    val_14253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 32), 'val', False)
    # Processing the call keyword arguments (line 225)
    kwargs_14254 = {}
    # Getting the type of '_np' (line 225)
    _np_14251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), '_np', False)
    # Obtaining the member 'asanyarray' of a type (line 225)
    asanyarray_14252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 17), _np_14251, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 225)
    asanyarray_call_result_14255 = invoke(stypy.reporting.localization.Localization(__file__, 225, 17), asanyarray_14252, *[val_14253], **kwargs_14254)
    
    float_14256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 39), 'float')
    # Applying the binary operator '-' (line 225)
    result_sub_14257 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 17), '-', asanyarray_call_result_14255, float_14256)
    
    float_14258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 46), 'float')
    # Applying the binary operator '*' (line 225)
    result_mul_14259 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 16), '*', result_sub_14257, float_14258)
    
    float_14260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 51), 'float')
    # Applying the binary operator 'div' (line 225)
    result_div_14261 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 49), 'div', result_mul_14259, float_14260)
    
    # Getting the type of 'zero_Celsius' (line 225)
    zero_Celsius_14262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 56), 'zero_Celsius')
    # Applying the binary operator '+' (line 225)
    result_add_14263 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 16), '+', result_div_14261, zero_Celsius_14262)
    
    # Assigning a type to the variable 'tempo' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tempo', result_add_14263)
    # SSA branch for the else part of an if statement (line 224)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 226)
    # Processing the call keyword arguments (line 226)
    kwargs_14266 = {}
    # Getting the type of 'old_scale' (line 226)
    old_scale_14264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 9), 'old_scale', False)
    # Obtaining the member 'lower' of a type (line 226)
    lower_14265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 9), old_scale_14264, 'lower')
    # Calling lower(args, kwargs) (line 226)
    lower_call_result_14267 = invoke(stypy.reporting.localization.Localization(__file__, 226, 9), lower_14265, *[], **kwargs_14266)
    
    
    # Obtaining an instance of the builtin type 'list' (line 226)
    list_14268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 226)
    # Adding element type (line 226)
    str_14269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 31), 'str', 'rankine')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 30), list_14268, str_14269)
    # Adding element type (line 226)
    str_14270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 42), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 30), list_14268, str_14270)
    
    # Applying the binary operator 'in' (line 226)
    result_contains_14271 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 9), 'in', lower_call_result_14267, list_14268)
    
    # Testing the type of an if condition (line 226)
    if_condition_14272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 9), result_contains_14271)
    # Assigning a type to the variable 'if_condition_14272' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 9), 'if_condition_14272', if_condition_14272)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 227):
    
    # Call to asanyarray(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'val' (line 227)
    val_14275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 31), 'val', False)
    # Processing the call keyword arguments (line 227)
    kwargs_14276 = {}
    # Getting the type of '_np' (line 227)
    _np_14273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), '_np', False)
    # Obtaining the member 'asanyarray' of a type (line 227)
    asanyarray_14274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), _np_14273, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 227)
    asanyarray_call_result_14277 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), asanyarray_14274, *[val_14275], **kwargs_14276)
    
    float_14278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 38), 'float')
    # Applying the binary operator '*' (line 227)
    result_mul_14279 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 16), '*', asanyarray_call_result_14277, float_14278)
    
    float_14280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 43), 'float')
    # Applying the binary operator 'div' (line 227)
    result_div_14281 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 41), 'div', result_mul_14279, float_14280)
    
    # Assigning a type to the variable 'tempo' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tempo', result_div_14281)
    # SSA branch for the else part of an if statement (line 226)
    module_type_store.open_ssa_branch('else')
    
    # Call to NotImplementedError(...): (line 229)
    # Processing the call arguments (line 229)
    str_14283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 34), 'str', '%s scale is unsupported: supported scales are Celsius, Kelvin, Fahrenheit and Rankine')
    # Getting the type of 'old_scale' (line 231)
    old_scale_14284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 46), 'old_scale', False)
    # Applying the binary operator '%' (line 229)
    result_mod_14285 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 34), '%', str_14283, old_scale_14284)
    
    # Processing the call keyword arguments (line 229)
    kwargs_14286 = {}
    # Getting the type of 'NotImplementedError' (line 229)
    NotImplementedError_14282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 229)
    NotImplementedError_call_result_14287 = invoke(stypy.reporting.localization.Localization(__file__, 229, 14), NotImplementedError_14282, *[result_mod_14285], **kwargs_14286)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 229, 8), NotImplementedError_call_result_14287, 'raise parameter', BaseException)
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 224)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 222)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 220)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to lower(...): (line 233)
    # Processing the call keyword arguments (line 233)
    kwargs_14290 = {}
    # Getting the type of 'new_scale' (line 233)
    new_scale_14288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 7), 'new_scale', False)
    # Obtaining the member 'lower' of a type (line 233)
    lower_14289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 7), new_scale_14288, 'lower')
    # Calling lower(args, kwargs) (line 233)
    lower_call_result_14291 = invoke(stypy.reporting.localization.Localization(__file__, 233, 7), lower_14289, *[], **kwargs_14290)
    
    
    # Obtaining an instance of the builtin type 'list' (line 233)
    list_14292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 233)
    # Adding element type (line 233)
    str_14293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'str', 'celsius')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 28), list_14292, str_14293)
    # Adding element type (line 233)
    str_14294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 40), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 28), list_14292, str_14294)
    
    # Applying the binary operator 'in' (line 233)
    result_contains_14295 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 7), 'in', lower_call_result_14291, list_14292)
    
    # Testing the type of an if condition (line 233)
    if_condition_14296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 4), result_contains_14295)
    # Assigning a type to the variable 'if_condition_14296' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'if_condition_14296', if_condition_14296)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 234):
    # Getting the type of 'tempo' (line 234)
    tempo_14297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 14), 'tempo')
    # Getting the type of 'zero_Celsius' (line 234)
    zero_Celsius_14298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'zero_Celsius')
    # Applying the binary operator '-' (line 234)
    result_sub_14299 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 14), '-', tempo_14297, zero_Celsius_14298)
    
    # Assigning a type to the variable 'res' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'res', result_sub_14299)
    # SSA branch for the else part of an if statement (line 233)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 235)
    # Processing the call keyword arguments (line 235)
    kwargs_14302 = {}
    # Getting the type of 'new_scale' (line 235)
    new_scale_14300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 9), 'new_scale', False)
    # Obtaining the member 'lower' of a type (line 235)
    lower_14301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 9), new_scale_14300, 'lower')
    # Calling lower(args, kwargs) (line 235)
    lower_call_result_14303 = invoke(stypy.reporting.localization.Localization(__file__, 235, 9), lower_14301, *[], **kwargs_14302)
    
    
    # Obtaining an instance of the builtin type 'list' (line 235)
    list_14304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 235)
    # Adding element type (line 235)
    str_14305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 31), 'str', 'kelvin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 30), list_14304, str_14305)
    # Adding element type (line 235)
    str_14306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 41), 'str', 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 30), list_14304, str_14306)
    
    # Applying the binary operator 'in' (line 235)
    result_contains_14307 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 9), 'in', lower_call_result_14303, list_14304)
    
    # Testing the type of an if condition (line 235)
    if_condition_14308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 9), result_contains_14307)
    # Assigning a type to the variable 'if_condition_14308' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 9), 'if_condition_14308', if_condition_14308)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 236):
    # Getting the type of 'tempo' (line 236)
    tempo_14309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'tempo')
    # Assigning a type to the variable 'res' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'res', tempo_14309)
    # SSA branch for the else part of an if statement (line 235)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_14312 = {}
    # Getting the type of 'new_scale' (line 237)
    new_scale_14310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 9), 'new_scale', False)
    # Obtaining the member 'lower' of a type (line 237)
    lower_14311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 9), new_scale_14310, 'lower')
    # Calling lower(args, kwargs) (line 237)
    lower_call_result_14313 = invoke(stypy.reporting.localization.Localization(__file__, 237, 9), lower_14311, *[], **kwargs_14312)
    
    
    # Obtaining an instance of the builtin type 'list' (line 237)
    list_14314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 237)
    # Adding element type (line 237)
    str_14315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 31), 'str', 'fahrenheit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 30), list_14314, str_14315)
    # Adding element type (line 237)
    str_14316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 45), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 30), list_14314, str_14316)
    
    # Applying the binary operator 'in' (line 237)
    result_contains_14317 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 9), 'in', lower_call_result_14313, list_14314)
    
    # Testing the type of an if condition (line 237)
    if_condition_14318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 9), result_contains_14317)
    # Assigning a type to the variable 'if_condition_14318' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 9), 'if_condition_14318', if_condition_14318)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 238):
    # Getting the type of 'tempo' (line 238)
    tempo_14319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'tempo')
    # Getting the type of 'zero_Celsius' (line 238)
    zero_Celsius_14320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'zero_Celsius')
    # Applying the binary operator '-' (line 238)
    result_sub_14321 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 15), '-', tempo_14319, zero_Celsius_14320)
    
    float_14322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 39), 'float')
    # Applying the binary operator '*' (line 238)
    result_mul_14323 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 14), '*', result_sub_14321, float_14322)
    
    float_14324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 44), 'float')
    # Applying the binary operator 'div' (line 238)
    result_div_14325 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 42), 'div', result_mul_14323, float_14324)
    
    float_14326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 49), 'float')
    # Applying the binary operator '+' (line 238)
    result_add_14327 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 14), '+', result_div_14325, float_14326)
    
    # Assigning a type to the variable 'res' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'res', result_add_14327)
    # SSA branch for the else part of an if statement (line 237)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 239)
    # Processing the call keyword arguments (line 239)
    kwargs_14330 = {}
    # Getting the type of 'new_scale' (line 239)
    new_scale_14328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 9), 'new_scale', False)
    # Obtaining the member 'lower' of a type (line 239)
    lower_14329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 9), new_scale_14328, 'lower')
    # Calling lower(args, kwargs) (line 239)
    lower_call_result_14331 = invoke(stypy.reporting.localization.Localization(__file__, 239, 9), lower_14329, *[], **kwargs_14330)
    
    
    # Obtaining an instance of the builtin type 'list' (line 239)
    list_14332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 239)
    # Adding element type (line 239)
    str_14333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 31), 'str', 'rankine')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 30), list_14332, str_14333)
    # Adding element type (line 239)
    str_14334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 42), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 30), list_14332, str_14334)
    
    # Applying the binary operator 'in' (line 239)
    result_contains_14335 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 9), 'in', lower_call_result_14331, list_14332)
    
    # Testing the type of an if condition (line 239)
    if_condition_14336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 9), result_contains_14335)
    # Assigning a type to the variable 'if_condition_14336' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 9), 'if_condition_14336', if_condition_14336)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 240):
    # Getting the type of 'tempo' (line 240)
    tempo_14337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'tempo')
    float_14338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'float')
    # Applying the binary operator '*' (line 240)
    result_mul_14339 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 14), '*', tempo_14337, float_14338)
    
    float_14340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'float')
    # Applying the binary operator 'div' (line 240)
    result_div_14341 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 25), 'div', result_mul_14339, float_14340)
    
    # Assigning a type to the variable 'res' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'res', result_div_14341)
    # SSA branch for the else part of an if statement (line 239)
    module_type_store.open_ssa_branch('else')
    
    # Call to NotImplementedError(...): (line 242)
    # Processing the call arguments (line 242)
    str_14343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'str', "'%s' scale is unsupported: supported scales are 'Celsius', 'Kelvin', 'Fahrenheit' and 'Rankine'")
    # Getting the type of 'new_scale' (line 244)
    new_scale_14344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 65), 'new_scale', False)
    # Applying the binary operator '%' (line 242)
    result_mod_14345 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 34), '%', str_14343, new_scale_14344)
    
    # Processing the call keyword arguments (line 242)
    kwargs_14346 = {}
    # Getting the type of 'NotImplementedError' (line 242)
    NotImplementedError_14342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 242)
    NotImplementedError_call_result_14347 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), NotImplementedError_14342, *[result_mod_14345], **kwargs_14346)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 242, 8), NotImplementedError_call_result_14347, 'raise parameter', BaseException)
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 246)
    res_14348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type', res_14348)
    
    # ################# End of 'convert_temperature(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convert_temperature' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_14349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14349)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convert_temperature'
    return stypy_return_type_14349

# Assigning a type to the variable 'convert_temperature' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'convert_temperature', convert_temperature)

@norecursion
def lambda2nu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lambda2nu'
    module_type_store = module_type_store.open_function_context('lambda2nu', 252, 0, False)
    
    # Passed parameters checking function
    lambda2nu.stypy_localization = localization
    lambda2nu.stypy_type_of_self = None
    lambda2nu.stypy_type_store = module_type_store
    lambda2nu.stypy_function_name = 'lambda2nu'
    lambda2nu.stypy_param_names_list = ['lambda_']
    lambda2nu.stypy_varargs_param_name = None
    lambda2nu.stypy_kwargs_param_name = None
    lambda2nu.stypy_call_defaults = defaults
    lambda2nu.stypy_call_varargs = varargs
    lambda2nu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lambda2nu', ['lambda_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lambda2nu', localization, ['lambda_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lambda2nu(...)' code ##################

    str_14350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, (-1)), 'str', '\n    Convert wavelength to optical frequency\n\n    Parameters\n    ----------\n    lambda_ : array_like\n        Wavelength(s) to be converted.\n\n    Returns\n    -------\n    nu : float or array of floats\n        Equivalent optical frequency.\n\n    Notes\n    -----\n    Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the\n    (vacuum) speed of light in meters/second.\n\n    Examples\n    --------\n    >>> from scipy.constants import lambda2nu, speed_of_light\n    >>> lambda2nu(np.array((1, speed_of_light)))\n    array([  2.99792458e+08,   1.00000000e+00])\n\n    ')
    
    # Call to asanyarray(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'c' (line 278)
    c_14353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'c', False)
    # Processing the call keyword arguments (line 278)
    kwargs_14354 = {}
    # Getting the type of '_np' (line 278)
    _np_14351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), '_np', False)
    # Obtaining the member 'asanyarray' of a type (line 278)
    asanyarray_14352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 11), _np_14351, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 278)
    asanyarray_call_result_14355 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), asanyarray_14352, *[c_14353], **kwargs_14354)
    
    # Getting the type of 'lambda_' (line 278)
    lambda__14356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 31), 'lambda_')
    # Applying the binary operator 'div' (line 278)
    result_div_14357 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 11), 'div', asanyarray_call_result_14355, lambda__14356)
    
    # Assigning a type to the variable 'stypy_return_type' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type', result_div_14357)
    
    # ################# End of 'lambda2nu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lambda2nu' in the type store
    # Getting the type of 'stypy_return_type' (line 252)
    stypy_return_type_14358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14358)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lambda2nu'
    return stypy_return_type_14358

# Assigning a type to the variable 'lambda2nu' (line 252)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'lambda2nu', lambda2nu)

@norecursion
def nu2lambda(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nu2lambda'
    module_type_store = module_type_store.open_function_context('nu2lambda', 281, 0, False)
    
    # Passed parameters checking function
    nu2lambda.stypy_localization = localization
    nu2lambda.stypy_type_of_self = None
    nu2lambda.stypy_type_store = module_type_store
    nu2lambda.stypy_function_name = 'nu2lambda'
    nu2lambda.stypy_param_names_list = ['nu']
    nu2lambda.stypy_varargs_param_name = None
    nu2lambda.stypy_kwargs_param_name = None
    nu2lambda.stypy_call_defaults = defaults
    nu2lambda.stypy_call_varargs = varargs
    nu2lambda.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nu2lambda', ['nu'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nu2lambda', localization, ['nu'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nu2lambda(...)' code ##################

    str_14359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, (-1)), 'str', '\n    Convert optical frequency to wavelength.\n\n    Parameters\n    ----------\n    nu : array_like\n        Optical frequency to be converted.\n\n    Returns\n    -------\n    lambda : float or array of floats\n        Equivalent wavelength(s).\n\n    Notes\n    -----\n    Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the\n    (vacuum) speed of light in meters/second.\n\n    Examples\n    --------\n    >>> from scipy.constants import nu2lambda, speed_of_light\n    >>> nu2lambda(np.array((1, speed_of_light)))\n    array([  2.99792458e+08,   1.00000000e+00])\n\n    ')
    # Getting the type of 'c' (line 307)
    c_14360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'c')
    
    # Call to asanyarray(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'nu' (line 307)
    nu_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), 'nu', False)
    # Processing the call keyword arguments (line 307)
    kwargs_14364 = {}
    # Getting the type of '_np' (line 307)
    _np_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), '_np', False)
    # Obtaining the member 'asanyarray' of a type (line 307)
    asanyarray_14362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), _np_14361, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 307)
    asanyarray_call_result_14365 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), asanyarray_14362, *[nu_14363], **kwargs_14364)
    
    # Applying the binary operator 'div' (line 307)
    result_div_14366 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 11), 'div', c_14360, asanyarray_call_result_14365)
    
    # Assigning a type to the variable 'stypy_return_type' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type', result_div_14366)
    
    # ################# End of 'nu2lambda(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nu2lambda' in the type store
    # Getting the type of 'stypy_return_type' (line 281)
    stypy_return_type_14367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14367)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nu2lambda'
    return stypy_return_type_14367

# Assigning a type to the variable 'nu2lambda' (line 281)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'nu2lambda', nu2lambda)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
