
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: r'''
2: ==================================
3: Constants (:mod:`scipy.constants`)
4: ==================================
5: 
6: .. currentmodule:: scipy.constants
7: 
8: Physical and mathematical constants and units.
9: 
10: 
11: Mathematical constants
12: ======================
13: 
14: ================  =================================================================
15: ``pi``            Pi
16: ``golden``        Golden ratio
17: ``golden_ratio``  Golden ratio
18: ================  =================================================================
19: 
20: 
21: Physical constants
22: ==================
23: 
24: ===========================  =================================================================
25: ``c``                        speed of light in vacuum
26: ``speed_of_light``           speed of light in vacuum
27: ``mu_0``                     the magnetic constant :math:`\mu_0`
28: ``epsilon_0``                the electric constant (vacuum permittivity), :math:`\epsilon_0`
29: ``h``                        the Planck constant :math:`h`
30: ``Planck``                   the Planck constant :math:`h`
31: ``hbar``                     :math:`\hbar = h/(2\pi)`
32: ``G``                        Newtonian constant of gravitation
33: ``gravitational_constant``   Newtonian constant of gravitation
34: ``g``                        standard acceleration of gravity
35: ``e``                        elementary charge
36: ``elementary_charge``        elementary charge
37: ``R``                        molar gas constant
38: ``gas_constant``             molar gas constant
39: ``alpha``                    fine-structure constant
40: ``fine_structure``           fine-structure constant
41: ``N_A``                      Avogadro constant
42: ``Avogadro``                 Avogadro constant
43: ``k``                        Boltzmann constant
44: ``Boltzmann``                Boltzmann constant
45: ``sigma``                    Stefan-Boltzmann constant :math:`\sigma`
46: ``Stefan_Boltzmann``         Stefan-Boltzmann constant :math:`\sigma`
47: ``Wien``                     Wien displacement law constant
48: ``Rydberg``                  Rydberg constant
49: ``m_e``                      electron mass
50: ``electron_mass``            electron mass
51: ``m_p``                      proton mass
52: ``proton_mass``              proton mass
53: ``m_n``                      neutron mass
54: ``neutron_mass``             neutron mass
55: ===========================  =================================================================
56: 
57: 
58: Constants database
59: ------------------
60: 
61: In addition to the above variables, :mod:`scipy.constants` also contains the
62: 2014 CODATA recommended values [CODATA2014]_ database containing more physical
63: constants.
64: 
65: .. autosummary::
66:    :toctree: generated/
67: 
68:    value      -- Value in physical_constants indexed by key
69:    unit       -- Unit in physical_constants indexed by key
70:    precision  -- Relative precision in physical_constants indexed by key
71:    find       -- Return list of physical_constant keys with a given string
72:    ConstantWarning -- Constant sought not in newest CODATA data set
73: 
74: .. data:: physical_constants
75: 
76:    Dictionary of physical constants, of the format
77:    ``physical_constants[name] = (value, unit, uncertainty)``.
78: 
79: Available constants:
80: 
81: ======================================================================  ====
82: %(constant_names)s
83: ======================================================================  ====
84: 
85: 
86: Units
87: =====
88: 
89: SI prefixes
90: -----------
91: 
92: ============  =================================================================
93: ``yotta``     :math:`10^{24}`
94: ``zetta``     :math:`10^{21}`
95: ``exa``       :math:`10^{18}`
96: ``peta``      :math:`10^{15}`
97: ``tera``      :math:`10^{12}`
98: ``giga``      :math:`10^{9}`
99: ``mega``      :math:`10^{6}`
100: ``kilo``      :math:`10^{3}`
101: ``hecto``     :math:`10^{2}`
102: ``deka``      :math:`10^{1}`
103: ``deci``      :math:`10^{-1}`
104: ``centi``     :math:`10^{-2}`
105: ``milli``     :math:`10^{-3}`
106: ``micro``     :math:`10^{-6}`
107: ``nano``      :math:`10^{-9}`
108: ``pico``      :math:`10^{-12}`
109: ``femto``     :math:`10^{-15}`
110: ``atto``      :math:`10^{-18}`
111: ``zepto``     :math:`10^{-21}`
112: ============  =================================================================
113: 
114: Binary prefixes
115: ---------------
116: 
117: ============  =================================================================
118: ``kibi``      :math:`2^{10}`
119: ``mebi``      :math:`2^{20}`
120: ``gibi``      :math:`2^{30}`
121: ``tebi``      :math:`2^{40}`
122: ``pebi``      :math:`2^{50}`
123: ``exbi``      :math:`2^{60}`
124: ``zebi``      :math:`2^{70}`
125: ``yobi``      :math:`2^{80}`
126: ============  =================================================================
127: 
128: Mass
129: ----
130: 
131: =================  ============================================================
132: ``gram``           :math:`10^{-3}` kg
133: ``metric_ton``     :math:`10^{3}` kg
134: ``grain``          one grain in kg
135: ``lb``             one pound (avoirdupous) in kg
136: ``pound``          one pound (avoirdupous) in kg
137: ``blob``           one inch version of a slug in kg (added in 1.0.0)
138: ``slinch``         one inch version of a slug in kg (added in 1.0.0)
139: ``slug``           one slug in kg (added in 1.0.0)
140: ``oz``             one ounce in kg
141: ``ounce``          one ounce in kg
142: ``stone``          one stone in kg
143: ``grain``          one grain in kg
144: ``long_ton``       one long ton in kg
145: ``short_ton``      one short ton in kg
146: ``troy_ounce``     one Troy ounce in kg
147: ``troy_pound``     one Troy pound in kg
148: ``carat``          one carat in kg
149: ``m_u``            atomic mass constant (in kg)
150: ``u``              atomic mass constant (in kg)
151: ``atomic_mass``    atomic mass constant (in kg)
152: =================  ============================================================
153: 
154: Angle
155: -----
156: 
157: =================  ============================================================
158: ``degree``         degree in radians
159: ``arcmin``         arc minute in radians
160: ``arcminute``      arc minute in radians
161: ``arcsec``         arc second in radians
162: ``arcsecond``      arc second in radians
163: =================  ============================================================
164: 
165: 
166: Time
167: ----
168: 
169: =================  ============================================================
170: ``minute``         one minute in seconds
171: ``hour``           one hour in seconds
172: ``day``            one day in seconds
173: ``week``           one week in seconds
174: ``year``           one year (365 days) in seconds
175: ``Julian_year``    one Julian year (365.25 days) in seconds
176: =================  ============================================================
177: 
178: 
179: Length
180: ------
181: 
182: =====================  ============================================================
183: ``inch``               one inch in meters
184: ``foot``               one foot in meters
185: ``yard``               one yard in meters
186: ``mile``               one mile in meters
187: ``mil``                one mil in meters
188: ``pt``                 one point in meters
189: ``point``              one point in meters
190: ``survey_foot``        one survey foot in meters
191: ``survey_mile``        one survey mile in meters
192: ``nautical_mile``      one nautical mile in meters
193: ``fermi``              one Fermi in meters
194: ``angstrom``           one Angstrom in meters
195: ``micron``             one micron in meters
196: ``au``                 one astronomical unit in meters
197: ``astronomical_unit``  one astronomical unit in meters
198: ``light_year``         one light year in meters
199: ``parsec``             one parsec in meters
200: =====================  ============================================================
201: 
202: Pressure
203: --------
204: 
205: =================  ============================================================
206: ``atm``            standard atmosphere in pascals
207: ``atmosphere``     standard atmosphere in pascals
208: ``bar``            one bar in pascals
209: ``torr``           one torr (mmHg) in pascals
210: ``mmHg``           one torr (mmHg) in pascals
211: ``psi``            one psi in pascals
212: =================  ============================================================
213: 
214: Area
215: ----
216: 
217: =================  ============================================================
218: ``hectare``        one hectare in square meters
219: ``acre``           one acre in square meters
220: =================  ============================================================
221: 
222: 
223: Volume
224: ------
225: 
226: ===================    ========================================================
227: ``liter``              one liter in cubic meters
228: ``litre``              one liter in cubic meters
229: ``gallon``             one gallon (US) in cubic meters
230: ``gallon_US``          one gallon (US) in cubic meters
231: ``gallon_imp``         one gallon (UK) in cubic meters
232: ``fluid_ounce``        one fluid ounce (US) in cubic meters
233: ``fluid_ounce_US``     one fluid ounce (US) in cubic meters
234: ``fluid_ounce_imp``    one fluid ounce (UK) in cubic meters
235: ``bbl``                one barrel in cubic meters
236: ``barrel``             one barrel in cubic meters
237: ===================    ========================================================
238: 
239: Speed
240: -----
241: 
242: ==================    ==========================================================
243: ``kmh``               kilometers per hour in meters per second
244: ``mph``               miles per hour in meters per second
245: ``mach``              one Mach (approx., at 15 C, 1 atm) in meters per second
246: ``speed_of_sound``    one Mach (approx., at 15 C, 1 atm) in meters per second
247: ``knot``              one knot in meters per second
248: ==================    ==========================================================
249: 
250: 
251: Temperature
252: -----------
253: 
254: =====================  =======================================================
255: ``zero_Celsius``       zero of Celsius scale in Kelvin
256: ``degree_Fahrenheit``  one Fahrenheit (only differences) in Kelvins
257: =====================  =======================================================
258: 
259: .. autosummary::
260:    :toctree: generated/
261: 
262:    convert_temperature
263: 
264: Energy
265: ------
266: 
267: ====================  =======================================================
268: ``eV``                one electron volt in Joules
269: ``electron_volt``     one electron volt in Joules
270: ``calorie``           one calorie (thermochemical) in Joules
271: ``calorie_th``        one calorie (thermochemical) in Joules
272: ``calorie_IT``        one calorie (International Steam Table calorie, 1956) in Joules
273: ``erg``               one erg in Joules
274: ``Btu``               one British thermal unit (International Steam Table) in Joules
275: ``Btu_IT``            one British thermal unit (International Steam Table) in Joules
276: ``Btu_th``            one British thermal unit (thermochemical) in Joules
277: ``ton_TNT``           one ton of TNT in Joules
278: ====================  =======================================================
279: 
280: Power
281: -----
282: 
283: ====================  =======================================================
284: ``hp``                one horsepower in watts
285: ``horsepower``        one horsepower in watts
286: ====================  =======================================================
287: 
288: Force
289: -----
290: 
291: ====================  =======================================================
292: ``dyn``               one dyne in newtons
293: ``dyne``              one dyne in newtons
294: ``lbf``               one pound force in newtons
295: ``pound_force``       one pound force in newtons
296: ``kgf``               one kilogram force in newtons
297: ``kilogram_force``    one kilogram force in newtons
298: ====================  =======================================================
299: 
300: Optics
301: ------
302: 
303: .. autosummary::
304:    :toctree: generated/
305: 
306:    lambda2nu
307:    nu2lambda
308: 
309: References
310: ==========
311: 
312: .. [CODATA2014] CODATA Recommended Values of the Fundamental
313:    Physical Constants 2014.
314: 
315:    http://physics.nist.gov/cuu/Constants/index.html
316: 
317: '''
318: from __future__ import division, print_function, absolute_import
319: 
320: # Modules contributed by BasSw (wegwerp@gmail.com)
321: from .codata import *
322: from .constants import *
323: from .codata import _obsolete_constants
324: 
325: _constant_names = [(_k.lower(), _k, _v)
326:                    for _k, _v in physical_constants.items()
327:                    if _k not in _obsolete_constants]
328: _constant_names = "\n".join(["``%s``%s  %s %s" % (_x[1], " "*(66-len(_x[1])),
329:                                                   _x[2][0], _x[2][1])
330:                              for _x in sorted(_constant_names)])
331: if __doc__ is not None:
332:     __doc__ = __doc__ % dict(constant_names=_constant_names)
333: 
334: del _constant_names
335: 
336: __all__ = [s for s in dir() if not s.startswith('_')]
337: 
338: from scipy._lib._testutils import PytestTester
339: test = PytestTester(__name__)
340: del PytestTester
341: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_14398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, (-1)), 'str', '\n==================================\nConstants (:mod:`scipy.constants`)\n==================================\n\n.. currentmodule:: scipy.constants\n\nPhysical and mathematical constants and units.\n\n\nMathematical constants\n======================\n\n================  =================================================================\n``pi``            Pi\n``golden``        Golden ratio\n``golden_ratio``  Golden ratio\n================  =================================================================\n\n\nPhysical constants\n==================\n\n===========================  =================================================================\n``c``                        speed of light in vacuum\n``speed_of_light``           speed of light in vacuum\n``mu_0``                     the magnetic constant :math:`\\mu_0`\n``epsilon_0``                the electric constant (vacuum permittivity), :math:`\\epsilon_0`\n``h``                        the Planck constant :math:`h`\n``Planck``                   the Planck constant :math:`h`\n``hbar``                     :math:`\\hbar = h/(2\\pi)`\n``G``                        Newtonian constant of gravitation\n``gravitational_constant``   Newtonian constant of gravitation\n``g``                        standard acceleration of gravity\n``e``                        elementary charge\n``elementary_charge``        elementary charge\n``R``                        molar gas constant\n``gas_constant``             molar gas constant\n``alpha``                    fine-structure constant\n``fine_structure``           fine-structure constant\n``N_A``                      Avogadro constant\n``Avogadro``                 Avogadro constant\n``k``                        Boltzmann constant\n``Boltzmann``                Boltzmann constant\n``sigma``                    Stefan-Boltzmann constant :math:`\\sigma`\n``Stefan_Boltzmann``         Stefan-Boltzmann constant :math:`\\sigma`\n``Wien``                     Wien displacement law constant\n``Rydberg``                  Rydberg constant\n``m_e``                      electron mass\n``electron_mass``            electron mass\n``m_p``                      proton mass\n``proton_mass``              proton mass\n``m_n``                      neutron mass\n``neutron_mass``             neutron mass\n===========================  =================================================================\n\n\nConstants database\n------------------\n\nIn addition to the above variables, :mod:`scipy.constants` also contains the\n2014 CODATA recommended values [CODATA2014]_ database containing more physical\nconstants.\n\n.. autosummary::\n   :toctree: generated/\n\n   value      -- Value in physical_constants indexed by key\n   unit       -- Unit in physical_constants indexed by key\n   precision  -- Relative precision in physical_constants indexed by key\n   find       -- Return list of physical_constant keys with a given string\n   ConstantWarning -- Constant sought not in newest CODATA data set\n\n.. data:: physical_constants\n\n   Dictionary of physical constants, of the format\n   ``physical_constants[name] = (value, unit, uncertainty)``.\n\nAvailable constants:\n\n======================================================================  ====\n%(constant_names)s\n======================================================================  ====\n\n\nUnits\n=====\n\nSI prefixes\n-----------\n\n============  =================================================================\n``yotta``     :math:`10^{24}`\n``zetta``     :math:`10^{21}`\n``exa``       :math:`10^{18}`\n``peta``      :math:`10^{15}`\n``tera``      :math:`10^{12}`\n``giga``      :math:`10^{9}`\n``mega``      :math:`10^{6}`\n``kilo``      :math:`10^{3}`\n``hecto``     :math:`10^{2}`\n``deka``      :math:`10^{1}`\n``deci``      :math:`10^{-1}`\n``centi``     :math:`10^{-2}`\n``milli``     :math:`10^{-3}`\n``micro``     :math:`10^{-6}`\n``nano``      :math:`10^{-9}`\n``pico``      :math:`10^{-12}`\n``femto``     :math:`10^{-15}`\n``atto``      :math:`10^{-18}`\n``zepto``     :math:`10^{-21}`\n============  =================================================================\n\nBinary prefixes\n---------------\n\n============  =================================================================\n``kibi``      :math:`2^{10}`\n``mebi``      :math:`2^{20}`\n``gibi``      :math:`2^{30}`\n``tebi``      :math:`2^{40}`\n``pebi``      :math:`2^{50}`\n``exbi``      :math:`2^{60}`\n``zebi``      :math:`2^{70}`\n``yobi``      :math:`2^{80}`\n============  =================================================================\n\nMass\n----\n\n=================  ============================================================\n``gram``           :math:`10^{-3}` kg\n``metric_ton``     :math:`10^{3}` kg\n``grain``          one grain in kg\n``lb``             one pound (avoirdupous) in kg\n``pound``          one pound (avoirdupous) in kg\n``blob``           one inch version of a slug in kg (added in 1.0.0)\n``slinch``         one inch version of a slug in kg (added in 1.0.0)\n``slug``           one slug in kg (added in 1.0.0)\n``oz``             one ounce in kg\n``ounce``          one ounce in kg\n``stone``          one stone in kg\n``grain``          one grain in kg\n``long_ton``       one long ton in kg\n``short_ton``      one short ton in kg\n``troy_ounce``     one Troy ounce in kg\n``troy_pound``     one Troy pound in kg\n``carat``          one carat in kg\n``m_u``            atomic mass constant (in kg)\n``u``              atomic mass constant (in kg)\n``atomic_mass``    atomic mass constant (in kg)\n=================  ============================================================\n\nAngle\n-----\n\n=================  ============================================================\n``degree``         degree in radians\n``arcmin``         arc minute in radians\n``arcminute``      arc minute in radians\n``arcsec``         arc second in radians\n``arcsecond``      arc second in radians\n=================  ============================================================\n\n\nTime\n----\n\n=================  ============================================================\n``minute``         one minute in seconds\n``hour``           one hour in seconds\n``day``            one day in seconds\n``week``           one week in seconds\n``year``           one year (365 days) in seconds\n``Julian_year``    one Julian year (365.25 days) in seconds\n=================  ============================================================\n\n\nLength\n------\n\n=====================  ============================================================\n``inch``               one inch in meters\n``foot``               one foot in meters\n``yard``               one yard in meters\n``mile``               one mile in meters\n``mil``                one mil in meters\n``pt``                 one point in meters\n``point``              one point in meters\n``survey_foot``        one survey foot in meters\n``survey_mile``        one survey mile in meters\n``nautical_mile``      one nautical mile in meters\n``fermi``              one Fermi in meters\n``angstrom``           one Angstrom in meters\n``micron``             one micron in meters\n``au``                 one astronomical unit in meters\n``astronomical_unit``  one astronomical unit in meters\n``light_year``         one light year in meters\n``parsec``             one parsec in meters\n=====================  ============================================================\n\nPressure\n--------\n\n=================  ============================================================\n``atm``            standard atmosphere in pascals\n``atmosphere``     standard atmosphere in pascals\n``bar``            one bar in pascals\n``torr``           one torr (mmHg) in pascals\n``mmHg``           one torr (mmHg) in pascals\n``psi``            one psi in pascals\n=================  ============================================================\n\nArea\n----\n\n=================  ============================================================\n``hectare``        one hectare in square meters\n``acre``           one acre in square meters\n=================  ============================================================\n\n\nVolume\n------\n\n===================    ========================================================\n``liter``              one liter in cubic meters\n``litre``              one liter in cubic meters\n``gallon``             one gallon (US) in cubic meters\n``gallon_US``          one gallon (US) in cubic meters\n``gallon_imp``         one gallon (UK) in cubic meters\n``fluid_ounce``        one fluid ounce (US) in cubic meters\n``fluid_ounce_US``     one fluid ounce (US) in cubic meters\n``fluid_ounce_imp``    one fluid ounce (UK) in cubic meters\n``bbl``                one barrel in cubic meters\n``barrel``             one barrel in cubic meters\n===================    ========================================================\n\nSpeed\n-----\n\n==================    ==========================================================\n``kmh``               kilometers per hour in meters per second\n``mph``               miles per hour in meters per second\n``mach``              one Mach (approx., at 15 C, 1 atm) in meters per second\n``speed_of_sound``    one Mach (approx., at 15 C, 1 atm) in meters per second\n``knot``              one knot in meters per second\n==================    ==========================================================\n\n\nTemperature\n-----------\n\n=====================  =======================================================\n``zero_Celsius``       zero of Celsius scale in Kelvin\n``degree_Fahrenheit``  one Fahrenheit (only differences) in Kelvins\n=====================  =======================================================\n\n.. autosummary::\n   :toctree: generated/\n\n   convert_temperature\n\nEnergy\n------\n\n====================  =======================================================\n``eV``                one electron volt in Joules\n``electron_volt``     one electron volt in Joules\n``calorie``           one calorie (thermochemical) in Joules\n``calorie_th``        one calorie (thermochemical) in Joules\n``calorie_IT``        one calorie (International Steam Table calorie, 1956) in Joules\n``erg``               one erg in Joules\n``Btu``               one British thermal unit (International Steam Table) in Joules\n``Btu_IT``            one British thermal unit (International Steam Table) in Joules\n``Btu_th``            one British thermal unit (thermochemical) in Joules\n``ton_TNT``           one ton of TNT in Joules\n====================  =======================================================\n\nPower\n-----\n\n====================  =======================================================\n``hp``                one horsepower in watts\n``horsepower``        one horsepower in watts\n====================  =======================================================\n\nForce\n-----\n\n====================  =======================================================\n``dyn``               one dyne in newtons\n``dyne``              one dyne in newtons\n``lbf``               one pound force in newtons\n``pound_force``       one pound force in newtons\n``kgf``               one kilogram force in newtons\n``kilogram_force``    one kilogram force in newtons\n====================  =======================================================\n\nOptics\n------\n\n.. autosummary::\n   :toctree: generated/\n\n   lambda2nu\n   nu2lambda\n\nReferences\n==========\n\n.. [CODATA2014] CODATA Recommended Values of the Fundamental\n   Physical Constants 2014.\n\n   http://physics.nist.gov/cuu/Constants/index.html\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 321, 0))

# 'from scipy.constants.codata import ' statement (line 321)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/')
import_14399 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 321, 0), 'scipy.constants.codata')

if (type(import_14399) is not StypyTypeError):

    if (import_14399 != 'pyd_module'):
        __import__(import_14399)
        sys_modules_14400 = sys.modules[import_14399]
        import_from_module(stypy.reporting.localization.Localization(__file__, 321, 0), 'scipy.constants.codata', sys_modules_14400.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 321, 0), __file__, sys_modules_14400, sys_modules_14400.module_type_store, module_type_store)
    else:
        from scipy.constants.codata import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 321, 0), 'scipy.constants.codata', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.constants.codata' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'scipy.constants.codata', import_14399)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 322, 0))

# 'from scipy.constants.constants import ' statement (line 322)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/')
import_14401 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 322, 0), 'scipy.constants.constants')

if (type(import_14401) is not StypyTypeError):

    if (import_14401 != 'pyd_module'):
        __import__(import_14401)
        sys_modules_14402 = sys.modules[import_14401]
        import_from_module(stypy.reporting.localization.Localization(__file__, 322, 0), 'scipy.constants.constants', sys_modules_14402.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 322, 0), __file__, sys_modules_14402, sys_modules_14402.module_type_store, module_type_store)
    else:
        from scipy.constants.constants import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 322, 0), 'scipy.constants.constants', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.constants.constants' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'scipy.constants.constants', import_14401)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 323, 0))

# 'from scipy.constants.codata import _obsolete_constants' statement (line 323)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/')
import_14403 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 323, 0), 'scipy.constants.codata')

if (type(import_14403) is not StypyTypeError):

    if (import_14403 != 'pyd_module'):
        __import__(import_14403)
        sys_modules_14404 = sys.modules[import_14403]
        import_from_module(stypy.reporting.localization.Localization(__file__, 323, 0), 'scipy.constants.codata', sys_modules_14404.module_type_store, module_type_store, ['_obsolete_constants'])
        nest_module(stypy.reporting.localization.Localization(__file__, 323, 0), __file__, sys_modules_14404, sys_modules_14404.module_type_store, module_type_store)
    else:
        from scipy.constants.codata import _obsolete_constants

        import_from_module(stypy.reporting.localization.Localization(__file__, 323, 0), 'scipy.constants.codata', None, module_type_store, ['_obsolete_constants'], [_obsolete_constants])

else:
    # Assigning a type to the variable 'scipy.constants.codata' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'scipy.constants.codata', import_14403)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/')


# Assigning a ListComp to a Name (line 325):
# Calculating list comprehension
# Calculating comprehension expression

# Call to items(...): (line 326)
# Processing the call keyword arguments (line 326)
kwargs_14417 = {}
# Getting the type of 'physical_constants' (line 326)
physical_constants_14415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 33), 'physical_constants', False)
# Obtaining the member 'items' of a type (line 326)
items_14416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 33), physical_constants_14415, 'items')
# Calling items(args, kwargs) (line 326)
items_call_result_14418 = invoke(stypy.reporting.localization.Localization(__file__, 326, 33), items_14416, *[], **kwargs_14417)

comprehension_14419 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 19), items_call_result_14418)
# Assigning a type to the variable '_k' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), '_k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 19), comprehension_14419))
# Assigning a type to the variable '_v' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), '_v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 19), comprehension_14419))

# Getting the type of '_k' (line 327)
_k_14412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 22), '_k')
# Getting the type of '_obsolete_constants' (line 327)
_obsolete_constants_14413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), '_obsolete_constants')
# Applying the binary operator 'notin' (line 327)
result_contains_14414 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 22), 'notin', _k_14412, _obsolete_constants_14413)


# Obtaining an instance of the builtin type 'tuple' (line 325)
tuple_14405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 325)
# Adding element type (line 325)

# Call to lower(...): (line 325)
# Processing the call keyword arguments (line 325)
kwargs_14408 = {}
# Getting the type of '_k' (line 325)
_k_14406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), '_k', False)
# Obtaining the member 'lower' of a type (line 325)
lower_14407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 20), _k_14406, 'lower')
# Calling lower(args, kwargs) (line 325)
lower_call_result_14409 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), lower_14407, *[], **kwargs_14408)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 20), tuple_14405, lower_call_result_14409)
# Adding element type (line 325)
# Getting the type of '_k' (line 325)
_k_14410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 32), '_k')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 20), tuple_14405, _k_14410)
# Adding element type (line 325)
# Getting the type of '_v' (line 325)
_v_14411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 36), '_v')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 20), tuple_14405, _v_14411)

list_14420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 19), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 19), list_14420, tuple_14405)
# Assigning a type to the variable '_constant_names' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), '_constant_names', list_14420)

# Assigning a Call to a Name (line 328):

# Call to join(...): (line 328)
# Processing the call arguments (line 328)
# Calculating list comprehension
# Calculating comprehension expression

# Call to sorted(...): (line 330)
# Processing the call arguments (line 330)
# Getting the type of '_constant_names' (line 330)
_constant_names_14456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 46), '_constant_names', False)
# Processing the call keyword arguments (line 330)
kwargs_14457 = {}
# Getting the type of 'sorted' (line 330)
sorted_14455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 39), 'sorted', False)
# Calling sorted(args, kwargs) (line 330)
sorted_call_result_14458 = invoke(stypy.reporting.localization.Localization(__file__, 330, 39), sorted_14455, *[_constant_names_14456], **kwargs_14457)

comprehension_14459 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 29), sorted_call_result_14458)
# Assigning a type to the variable '_x' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 29), '_x', comprehension_14459)
str_14423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 29), 'str', '``%s``%s  %s %s')

# Obtaining an instance of the builtin type 'tuple' (line 328)
tuple_14424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 328)
# Adding element type (line 328)

# Obtaining the type of the subscript
int_14425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 53), 'int')
# Getting the type of '_x' (line 328)
_x_14426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 50), '_x', False)
# Obtaining the member '__getitem__' of a type (line 328)
getitem___14427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 50), _x_14426, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 328)
subscript_call_result_14428 = invoke(stypy.reporting.localization.Localization(__file__, 328, 50), getitem___14427, int_14425)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 50), tuple_14424, subscript_call_result_14428)
# Adding element type (line 328)
str_14429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 57), 'str', ' ')
int_14430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 62), 'int')

# Call to len(...): (line 328)
# Processing the call arguments (line 328)

# Obtaining the type of the subscript
int_14432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 72), 'int')
# Getting the type of '_x' (line 328)
_x_14433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 69), '_x', False)
# Obtaining the member '__getitem__' of a type (line 328)
getitem___14434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 69), _x_14433, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 328)
subscript_call_result_14435 = invoke(stypy.reporting.localization.Localization(__file__, 328, 69), getitem___14434, int_14432)

# Processing the call keyword arguments (line 328)
kwargs_14436 = {}
# Getting the type of 'len' (line 328)
len_14431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 65), 'len', False)
# Calling len(args, kwargs) (line 328)
len_call_result_14437 = invoke(stypy.reporting.localization.Localization(__file__, 328, 65), len_14431, *[subscript_call_result_14435], **kwargs_14436)

# Applying the binary operator '-' (line 328)
result_sub_14438 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 62), '-', int_14430, len_call_result_14437)

# Applying the binary operator '*' (line 328)
result_mul_14439 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 57), '*', str_14429, result_sub_14438)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 50), tuple_14424, result_mul_14439)
# Adding element type (line 328)

# Obtaining the type of the subscript
int_14440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 56), 'int')

# Obtaining the type of the subscript
int_14441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 53), 'int')
# Getting the type of '_x' (line 329)
_x_14442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 50), '_x', False)
# Obtaining the member '__getitem__' of a type (line 329)
getitem___14443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 50), _x_14442, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 329)
subscript_call_result_14444 = invoke(stypy.reporting.localization.Localization(__file__, 329, 50), getitem___14443, int_14441)

# Obtaining the member '__getitem__' of a type (line 329)
getitem___14445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 50), subscript_call_result_14444, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 329)
subscript_call_result_14446 = invoke(stypy.reporting.localization.Localization(__file__, 329, 50), getitem___14445, int_14440)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 50), tuple_14424, subscript_call_result_14446)
# Adding element type (line 328)

# Obtaining the type of the subscript
int_14447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 66), 'int')

# Obtaining the type of the subscript
int_14448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 63), 'int')
# Getting the type of '_x' (line 329)
_x_14449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 60), '_x', False)
# Obtaining the member '__getitem__' of a type (line 329)
getitem___14450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 60), _x_14449, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 329)
subscript_call_result_14451 = invoke(stypy.reporting.localization.Localization(__file__, 329, 60), getitem___14450, int_14448)

# Obtaining the member '__getitem__' of a type (line 329)
getitem___14452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 60), subscript_call_result_14451, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 329)
subscript_call_result_14453 = invoke(stypy.reporting.localization.Localization(__file__, 329, 60), getitem___14452, int_14447)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 50), tuple_14424, subscript_call_result_14453)

# Applying the binary operator '%' (line 328)
result_mod_14454 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 29), '%', str_14423, tuple_14424)

list_14460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 29), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 29), list_14460, result_mod_14454)
# Processing the call keyword arguments (line 328)
kwargs_14461 = {}
str_14421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 18), 'str', '\n')
# Obtaining the member 'join' of a type (line 328)
join_14422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 18), str_14421, 'join')
# Calling join(args, kwargs) (line 328)
join_call_result_14462 = invoke(stypy.reporting.localization.Localization(__file__, 328, 18), join_14422, *[list_14460], **kwargs_14461)

# Assigning a type to the variable '_constant_names' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), '_constant_names', join_call_result_14462)

# Type idiom detected: calculating its left and rigth part (line 331)
# Getting the type of '__doc__' (line 331)
doc___14463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 0), '__doc__')
# Getting the type of 'None' (line 331)
None_14464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'None')

(may_be_14465, more_types_in_union_14466) = may_not_be_none(doc___14463, None_14464)

if may_be_14465:

    if more_types_in_union_14466:
        # Runtime conditional SSA (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a BinOp to a Name (line 332):
    # Getting the type of '__doc__' (line 332)
    doc___14467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), '__doc__')
    
    # Call to dict(...): (line 332)
    # Processing the call keyword arguments (line 332)
    # Getting the type of '_constant_names' (line 332)
    _constant_names_14469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 44), '_constant_names', False)
    keyword_14470 = _constant_names_14469
    kwargs_14471 = {'constant_names': keyword_14470}
    # Getting the type of 'dict' (line 332)
    dict_14468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'dict', False)
    # Calling dict(args, kwargs) (line 332)
    dict_call_result_14472 = invoke(stypy.reporting.localization.Localization(__file__, 332, 24), dict_14468, *[], **kwargs_14471)
    
    # Applying the binary operator '%' (line 332)
    result_mod_14473 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 14), '%', doc___14467, dict_call_result_14472)
    
    # Assigning a type to the variable '__doc__' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), '__doc__', result_mod_14473)

    if more_types_in_union_14466:
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()



# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 334, 0), module_type_store, '_constant_names')

# Assigning a ListComp to a Name (line 336):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 336)
# Processing the call keyword arguments (line 336)
kwargs_14482 = {}
# Getting the type of 'dir' (line 336)
dir_14481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 22), 'dir', False)
# Calling dir(args, kwargs) (line 336)
dir_call_result_14483 = invoke(stypy.reporting.localization.Localization(__file__, 336, 22), dir_14481, *[], **kwargs_14482)

comprehension_14484 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 11), dir_call_result_14483)
# Assigning a type to the variable 's' (line 336)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 's', comprehension_14484)


# Call to startswith(...): (line 336)
# Processing the call arguments (line 336)
str_14477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 48), 'str', '_')
# Processing the call keyword arguments (line 336)
kwargs_14478 = {}
# Getting the type of 's' (line 336)
s_14475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 336)
startswith_14476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 35), s_14475, 'startswith')
# Calling startswith(args, kwargs) (line 336)
startswith_call_result_14479 = invoke(stypy.reporting.localization.Localization(__file__, 336, 35), startswith_14476, *[str_14477], **kwargs_14478)

# Applying the 'not' unary operator (line 336)
result_not__14480 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 31), 'not', startswith_call_result_14479)

# Getting the type of 's' (line 336)
s_14474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 's')
list_14485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 11), list_14485, s_14474)
# Assigning a type to the variable '__all__' (line 336)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), '__all__', list_14485)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 338, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 338)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/constants/')
import_14486 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 338, 0), 'scipy._lib._testutils')

if (type(import_14486) is not StypyTypeError):

    if (import_14486 != 'pyd_module'):
        __import__(import_14486)
        sys_modules_14487 = sys.modules[import_14486]
        import_from_module(stypy.reporting.localization.Localization(__file__, 338, 0), 'scipy._lib._testutils', sys_modules_14487.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 338, 0), __file__, sys_modules_14487, sys_modules_14487.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 338, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 0), 'scipy._lib._testutils', import_14486)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/constants/')


# Assigning a Call to a Name (line 339):

# Call to PytestTester(...): (line 339)
# Processing the call arguments (line 339)
# Getting the type of '__name__' (line 339)
name___14489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 20), '__name__', False)
# Processing the call keyword arguments (line 339)
kwargs_14490 = {}
# Getting the type of 'PytestTester' (line 339)
PytestTester_14488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 339)
PytestTester_call_result_14491 = invoke(stypy.reporting.localization.Localization(__file__, 339, 7), PytestTester_14488, *[name___14489], **kwargs_14490)

# Assigning a type to the variable 'test' (line 339)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 0), 'test', PytestTester_call_result_14491)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 340, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
