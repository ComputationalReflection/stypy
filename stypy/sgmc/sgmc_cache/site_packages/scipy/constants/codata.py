
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Compiled by Charles Harris, dated October 3, 2002
2: # updated to 2002 values by BasSw, 2006
3: # Updated to 2006 values by Vincent Davis June 2010
4: # Updated to 2014 values by Joseph Booker, 2015
5: 
6: '''
7: Fundamental Physical Constants
8: ------------------------------
9: 
10: These constants are taken from CODATA Recommended Values of the Fundamental
11: Physical Constants 2014.
12: 
13: Object
14: ------
15: physical_constants : dict
16:     A dictionary containing physical constants. Keys are the names of physical
17:     constants, values are tuples (value, units, precision).
18: 
19: Functions
20: ---------
21: value(key):
22:     Returns the value of the physical constant(key).
23: unit(key):
24:     Returns the units of the physical constant(key).
25: precision(key):
26:     Returns the relative precision of the physical constant(key).
27: find(sub):
28:     Prints or returns list of keys containing the string sub, default is all.
29: 
30: Source
31: ------
32: The values of the constants provided at this site are recommended for
33: international use by CODATA and are the latest available. Termed the "2014
34: CODATA recommended values," they are generally recognized worldwide for use in
35: all fields of science and technology. The values became available on 25 June
36: 2015 and replaced the 2010 CODATA set. They are based on all of the data
37: available through 31 December 2014. The 2014 adjustment was carried out under
38: the auspices of the CODATA Task Group on Fundamental Constants. Also available
39: is an introduction to the constants for non-experts at
40: http://physics.nist.gov/cuu/Constants/introduction.html
41: 
42: References
43: ----------
44: Theoretical and experimental publications relevant to the fundamental constants
45: and closely related precision measurements published since the mid 1980s, but
46: also including many older papers of particular interest, some of which date
47: back to the 1800s. To search bibliography visit
48: 
49: http://physics.nist.gov/cuu/Constants/
50: 
51: '''
52: from __future__ import division, print_function, absolute_import
53: 
54: import warnings
55: from math import pi, sqrt
56: 
57: __all__ = ['physical_constants', 'value', 'unit', 'precision', 'find',
58:            'ConstantWarning']
59: 
60: '''
61: Source:  http://physics.nist.gov/cuu/Constants/index.html
62: 
63: The values of the constants provided at the above site are recommended for
64: international use by CODATA and are the latest available. Termed the "2006
65: CODATA recommended values", they are generally recognized worldwide for use
66: in all fields of science and technology. The values became available in March
67: 2007 and replaced the 2002 CODATA set. They are based on all of the data
68: available through 31 December 2006. The 2006 adjustment was carried out under
69: the auspices of the CODATA Task Group on Fundamental Constants.
70: '''
71: 
72: #
73: # Source:  http://physics.nist.gov/cuu/Constants/index.html
74: #
75: 
76: # Quantity                                             Value                 Uncertainty          Unit
77: # ---------------------------------------------------- --------------------- -------------------- -------------
78: txt2002 = '''\
79: Wien displacement law constant                         2.897 7685e-3         0.000 0051e-3         m K
80: atomic unit of 1st hyperpolarizablity                  3.206 361 51e-53      0.000 000 28e-53      C^3 m^3 J^-2
81: atomic unit of 2nd hyperpolarizablity                  6.235 3808e-65        0.000 0011e-65        C^4 m^4 J^-3
82: atomic unit of electric dipole moment                  8.478 353 09e-30      0.000 000 73e-30      C m
83: atomic unit of electric polarizablity                  1.648 777 274e-41     0.000 000 016e-41     C^2 m^2 J^-1
84: atomic unit of electric quadrupole moment              4.486 551 24e-40      0.000 000 39e-40      C m^2
85: atomic unit of magn. dipole moment                     1.854 801 90e-23      0.000 000 16e-23      J T^-1
86: atomic unit of magn. flux density                      2.350 517 42e5        0.000 000 20e5        T
87: deuteron magn. moment                                  0.433 073 482e-26     0.000 000 038e-26     J T^-1
88: deuteron magn. moment to Bohr magneton ratio           0.466 975 4567e-3     0.000 000 0050e-3
89: deuteron magn. moment to nuclear magneton ratio        0.857 438 2329        0.000 000 0092
90: deuteron-electron magn. moment ratio                   -4.664 345 548e-4     0.000 000 050e-4
91: deuteron-proton magn. moment ratio                     0.307 012 2084        0.000 000 0045
92: deuteron-neutron magn. moment ratio                    -0.448 206 52         0.000 000 11
93: electron gyromagn. ratio                               1.760 859 74e11       0.000 000 15e11       s^-1 T^-1
94: electron gyromagn. ratio over 2 pi                     28 024.9532           0.0024                MHz T^-1
95: electron magn. moment                                  -928.476 412e-26      0.000 080e-26         J T^-1
96: electron magn. moment to Bohr magneton ratio           -1.001 159 652 1859   0.000 000 000 0038
97: electron magn. moment to nuclear magneton ratio        -1838.281 971 07      0.000 000 85
98: electron magn. moment anomaly                          1.159 652 1859e-3     0.000 000 0038e-3
99: electron to shielded proton magn. moment ratio         -658.227 5956         0.000 0071
100: electron to shielded helion magn. moment ratio         864.058 255           0.000 010
101: electron-deuteron magn. moment ratio                   -2143.923 493         0.000 023
102: electron-muon magn. moment ratio                       206.766 9894          0.000 0054
103: electron-neutron magn. moment ratio                    960.920 50            0.000 23
104: electron-proton magn. moment ratio                     -658.210 6862         0.000 0066
105: magn. constant                                         12.566 370 614...e-7  0                     N A^-2
106: magn. flux quantum                                     2.067 833 72e-15      0.000 000 18e-15      Wb
107: muon magn. moment                                      -4.490 447 99e-26     0.000 000 40e-26      J T^-1
108: muon magn. moment to Bohr magneton ratio               -4.841 970 45e-3      0.000 000 13e-3
109: muon magn. moment to nuclear magneton ratio            -8.890 596 98         0.000 000 23
110: muon-proton magn. moment ratio                         -3.183 345 118        0.000 000 089
111: neutron gyromagn. ratio                                1.832 471 83e8        0.000 000 46e8        s^-1 T^-1
112: neutron gyromagn. ratio over 2 pi                      29.164 6950           0.000 0073            MHz T^-1
113: neutron magn. moment                                   -0.966 236 45e-26     0.000 000 24e-26      J T^-1
114: neutron magn. moment to Bohr magneton ratio            -1.041 875 63e-3      0.000 000 25e-3
115: neutron magn. moment to nuclear magneton ratio         -1.913 042 73         0.000 000 45
116: neutron to shielded proton magn. moment ratio          -0.684 996 94         0.000 000 16
117: neutron-electron magn. moment ratio                    1.040 668 82e-3       0.000 000 25e-3
118: neutron-proton magn. moment ratio                      -0.684 979 34         0.000 000 16
119: proton gyromagn. ratio                                 2.675 222 05e8        0.000 000 23e8        s^-1 T^-1
120: proton gyromagn. ratio over 2 pi                       42.577 4813           0.000 0037            MHz T^-1
121: proton magn. moment                                    1.410 606 71e-26      0.000 000 12e-26      J T^-1
122: proton magn. moment to Bohr magneton ratio             1.521 032 206e-3      0.000 000 015e-3
123: proton magn. moment to nuclear magneton ratio          2.792 847 351         0.000 000 028
124: proton magn. shielding correction                      25.689e-6             0.015e-6
125: proton-neutron magn. moment ratio                      -1.459 898 05         0.000 000 34
126: shielded helion gyromagn. ratio                        2.037 894 70e8        0.000 000 18e8        s^-1 T^-1
127: shielded helion gyromagn. ratio over 2 pi              32.434 1015           0.000 0028            MHz T^-1
128: shielded helion magn. moment                           -1.074 553 024e-26    0.000 000 093e-26     J T^-1
129: shielded helion magn. moment to Bohr magneton ratio    -1.158 671 474e-3     0.000 000 014e-3
130: shielded helion magn. moment to nuclear magneton ratio -2.127 497 723        0.000 000 025
131: shielded helion to proton magn. moment ratio           -0.761 766 562        0.000 000 012
132: shielded helion to shielded proton magn. moment ratio  -0.761 786 1313       0.000 000 0033
133: shielded helion gyromagn. ratio                        2.037 894 70e8        0.000 000 18e8        s^-1 T^-1
134: shielded helion gyromagn. ratio over 2 pi              32.434 1015           0.000 0028            MHz T^-1
135: shielded proton magn. moment                           1.410 570 47e-26      0.000 000 12e-26      J T^-1
136: shielded proton magn. moment to Bohr magneton ratio    1.520 993 132e-3      0.000 000 016e-3
137: shielded proton magn. moment to nuclear magneton ratio 2.792 775 604         0.000 000 030
138: {220} lattice spacing of silicon                       192.015 5965e-12      0.000 0070e-12        m'''
139: 
140: txt2006 = '''\
141: lattice spacing of silicon                             192.015 5762 e-12     0.000 0050 e-12       m
142: alpha particle-electron mass ratio                     7294.299 5365         0.000 0031
143: alpha particle mass                                    6.644 656 20 e-27     0.000 000 33 e-27     kg
144: alpha particle mass energy equivalent                  5.971 919 17 e-10     0.000 000 30 e-10     J
145: alpha particle mass energy equivalent in MeV           3727.379 109          0.000 093             MeV
146: alpha particle mass in u                               4.001 506 179 127     0.000 000 000 062     u
147: alpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 062 e-3 kg mol^-1
148: alpha particle-proton mass ratio                       3.972 599 689 51      0.000 000 000 41
149: Angstrom star                                          1.000 014 98 e-10     0.000 000 90 e-10     m
150: atomic mass constant                                   1.660 538 782 e-27    0.000 000 083 e-27    kg
151: atomic mass constant energy equivalent                 1.492 417 830 e-10    0.000 000 074 e-10    J
152: atomic mass constant energy equivalent in MeV          931.494 028           0.000 023             MeV
153: atomic mass unit-electron volt relationship            931.494 028 e6        0.000 023 e6          eV
154: atomic mass unit-hartree relationship                  3.423 177 7149 e7     0.000 000 0049 e7     E_h
155: atomic mass unit-hertz relationship                    2.252 342 7369 e23    0.000 000 0032 e23    Hz
156: atomic mass unit-inverse meter relationship            7.513 006 671 e14     0.000 000 011 e14     m^-1
157: atomic mass unit-joule relationship                    1.492 417 830 e-10    0.000 000 074 e-10    J
158: atomic mass unit-kelvin relationship                   1.080 9527 e13        0.000 0019 e13        K
159: atomic mass unit-kilogram relationship                 1.660 538 782 e-27    0.000 000 083 e-27    kg
160: atomic unit of 1st hyperpolarizability                 3.206 361 533 e-53    0.000 000 081 e-53    C^3 m^3 J^-2
161: atomic unit of 2nd hyperpolarizability                 6.235 380 95 e-65     0.000 000 31 e-65     C^4 m^4 J^-3
162: atomic unit of action                                  1.054 571 628 e-34    0.000 000 053 e-34    J s
163: atomic unit of charge                                  1.602 176 487 e-19    0.000 000 040 e-19    C
164: atomic unit of charge density                          1.081 202 300 e12     0.000 000 027 e12     C m^-3
165: atomic unit of current                                 6.623 617 63 e-3      0.000 000 17 e-3      A
166: atomic unit of electric dipole mom.                    8.478 352 81 e-30     0.000 000 21 e-30     C m
167: atomic unit of electric field                          5.142 206 32 e11      0.000 000 13 e11      V m^-1
168: atomic unit of electric field gradient                 9.717 361 66 e21      0.000 000 24 e21      V m^-2
169: atomic unit of electric polarizability                 1.648 777 2536 e-41   0.000 000 0034 e-41   C^2 m^2 J^-1
170: atomic unit of electric potential                      27.211 383 86         0.000 000 68          V
171: atomic unit of electric quadrupole mom.                4.486 551 07 e-40     0.000 000 11 e-40     C m^2
172: atomic unit of energy                                  4.359 743 94 e-18     0.000 000 22 e-18     J
173: atomic unit of force                                   8.238 722 06 e-8      0.000 000 41 e-8      N
174: atomic unit of length                                  0.529 177 208 59 e-10 0.000 000 000 36 e-10 m
175: atomic unit of mag. dipole mom.                        1.854 801 830 e-23    0.000 000 046 e-23    J T^-1
176: atomic unit of mag. flux density                       2.350 517 382 e5      0.000 000 059 e5      T
177: atomic unit of magnetizability                         7.891 036 433 e-29    0.000 000 027 e-29    J T^-2
178: atomic unit of mass                                    9.109 382 15 e-31     0.000 000 45 e-31     kg
179: atomic unit of momentum                                1.992 851 565 e-24    0.000 000 099 e-24    kg m s^-1
180: atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
181: atomic unit of time                                    2.418 884 326 505 e-17 0.000 000 000 016 e-17 s
182: atomic unit of velocity                                2.187 691 2541 e6     0.000 000 0015 e6     m s^-1
183: Avogadro constant                                      6.022 141 79 e23      0.000 000 30 e23      mol^-1
184: Bohr magneton                                          927.400 915 e-26      0.000 023 e-26        J T^-1
185: Bohr magneton in eV/T                                  5.788 381 7555 e-5    0.000 000 0079 e-5    eV T^-1
186: Bohr magneton in Hz/T                                  13.996 246 04 e9      0.000 000 35 e9       Hz T^-1
187: Bohr magneton in inverse meters per tesla              46.686 4515           0.000 0012            m^-1 T^-1
188: Bohr magneton in K/T                                   0.671 7131            0.000 0012            K T^-1
189: Bohr radius                                            0.529 177 208 59 e-10 0.000 000 000 36 e-10 m
190: Boltzmann constant                                     1.380 6504 e-23       0.000 0024 e-23       J K^-1
191: Boltzmann constant in eV/K                             8.617 343 e-5         0.000 015 e-5         eV K^-1
192: Boltzmann constant in Hz/K                             2.083 6644 e10        0.000 0036 e10        Hz K^-1
193: Boltzmann constant in inverse meters per kelvin        69.503 56             0.000 12              m^-1 K^-1
194: characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
195: classical electron radius                              2.817 940 2894 e-15   0.000 000 0058 e-15   m
196: Compton wavelength                                     2.426 310 2175 e-12   0.000 000 0033 e-12   m
197: Compton wavelength over 2 pi                           386.159 264 59 e-15   0.000 000 53 e-15     m
198: conductance quantum                                    7.748 091 7004 e-5    0.000 000 0053 e-5    S
199: conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
200: conventional value of von Klitzing constant            25 812.807            (exact)               ohm
201: Cu x unit                                              1.002 076 99 e-13     0.000 000 28 e-13     m
202: deuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4
203: deuteron-electron mass ratio                           3670.482 9654         0.000 0016
204: deuteron g factor                                      0.857 438 2308        0.000 000 0072
205: deuteron mag. mom.                                     0.433 073 465 e-26    0.000 000 011 e-26    J T^-1
206: deuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3
207: deuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072
208: deuteron mass                                          3.343 583 20 e-27     0.000 000 17 e-27     kg
209: deuteron mass energy equivalent                        3.005 062 72 e-10     0.000 000 15 e-10     J
210: deuteron mass energy equivalent in MeV                 1875.612 793          0.000 047             MeV
211: deuteron mass in u                                     2.013 553 212 724     0.000 000 000 078     u
212: deuteron molar mass                                    2.013 553 212 724 e-3 0.000 000 000 078 e-3 kg mol^-1
213: deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
214: deuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024
215: deuteron-proton mass ratio                             1.999 007 501 08      0.000 000 000 22
216: deuteron rms charge radius                             2.1402 e-15           0.0028 e-15           m
217: electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
218: electron charge to mass quotient                       -1.758 820 150 e11    0.000 000 044 e11     C kg^-1
219: electron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018
220: electron-deuteron mass ratio                           2.724 437 1093 e-4    0.000 000 0012 e-4
221: electron g factor                                      -2.002 319 304 3622   0.000 000 000 0015
222: electron gyromag. ratio                                1.760 859 770 e11     0.000 000 044 e11     s^-1 T^-1
223: electron gyromag. ratio over 2 pi                      28 024.953 64         0.000 70              MHz T^-1
224: electron mag. mom.                                     -928.476 377 e-26     0.000 023 e-26        J T^-1
225: electron mag. mom. anomaly                             1.159 652 181 11 e-3  0.000 000 000 74 e-3
226: electron mag. mom. to Bohr magneton ratio              -1.001 159 652 181 11 0.000 000 000 000 74
227: electron mag. mom. to nuclear magneton ratio           -1838.281 970 92      0.000 000 80
228: electron mass                                          9.109 382 15 e-31     0.000 000 45 e-31     kg
229: electron mass energy equivalent                        8.187 104 38 e-14     0.000 000 41 e-14     J
230: electron mass energy equivalent in MeV                 0.510 998 910         0.000 000 013         MeV
231: electron mass in u                                     5.485 799 0943 e-4    0.000 000 0023 e-4    u
232: electron molar mass                                    5.485 799 0943 e-7    0.000 000 0023 e-7    kg mol^-1
233: electron-muon mag. mom. ratio                          206.766 9877          0.000 0052
234: electron-muon mass ratio                               4.836 331 71 e-3      0.000 000 12 e-3
235: electron-neutron mag. mom. ratio                       960.920 50            0.000 23
236: electron-neutron mass ratio                            5.438 673 4459 e-4    0.000 000 0033 e-4
237: electron-proton mag. mom. ratio                        -658.210 6848         0.000 0054
238: electron-proton mass ratio                             5.446 170 2177 e-4    0.000 000 0024 e-4
239: electron-tau mass ratio                                2.875 64 e-4          0.000 47 e-4
240: electron to alpha particle mass ratio                  1.370 933 555 70 e-4  0.000 000 000 58 e-4
241: electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
242: electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
243: electron volt                                          1.602 176 487 e-19    0.000 000 040 e-19    J
244: electron volt-atomic mass unit relationship            1.073 544 188 e-9     0.000 000 027 e-9     u
245: electron volt-hartree relationship                     3.674 932 540 e-2     0.000 000 092 e-2     E_h
246: electron volt-hertz relationship                       2.417 989 454 e14     0.000 000 060 e14     Hz
247: electron volt-inverse meter relationship               8.065 544 65 e5       0.000 000 20 e5       m^-1
248: electron volt-joule relationship                       1.602 176 487 e-19    0.000 000 040 e-19    J
249: electron volt-kelvin relationship                      1.160 4505 e4         0.000 0020 e4         K
250: electron volt-kilogram relationship                    1.782 661 758 e-36    0.000 000 044 e-36    kg
251: elementary charge                                      1.602 176 487 e-19    0.000 000 040 e-19    C
252: elementary charge over h                               2.417 989 454 e14     0.000 000 060 e14     A J^-1
253: Faraday constant                                       96 485.3399           0.0024                C mol^-1
254: Faraday constant for conventional electric current     96 485.3401           0.0048                C_90 mol^-1
255: Fermi coupling constant                                1.166 37 e-5          0.000 01 e-5          GeV^-2
256: fine-structure constant                                7.297 352 5376 e-3    0.000 000 0050 e-3
257: first radiation constant                               3.741 771 18 e-16     0.000 000 19 e-16     W m^2
258: first radiation constant for spectral radiance         1.191 042 759 e-16    0.000 000 059 e-16    W m^2 sr^-1
259: hartree-atomic mass unit relationship                  2.921 262 2986 e-8    0.000 000 0042 e-8    u
260: hartree-electron volt relationship                     27.211 383 86         0.000 000 68          eV
261: Hartree energy                                         4.359 743 94 e-18     0.000 000 22 e-18     J
262: Hartree energy in eV                                   27.211 383 86         0.000 000 68          eV
263: hartree-hertz relationship                             6.579 683 920 722 e15 0.000 000 000 044 e15 Hz
264: hartree-inverse meter relationship                     2.194 746 313 705 e7  0.000 000 000 015 e7  m^-1
265: hartree-joule relationship                             4.359 743 94 e-18     0.000 000 22 e-18     J
266: hartree-kelvin relationship                            3.157 7465 e5         0.000 0055 e5         K
267: hartree-kilogram relationship                          4.850 869 34 e-35     0.000 000 24 e-35     kg
268: helion-electron mass ratio                             5495.885 2765         0.000 0052
269: helion mass                                            5.006 411 92 e-27     0.000 000 25 e-27     kg
270: helion mass energy equivalent                          4.499 538 64 e-10     0.000 000 22 e-10     J
271: helion mass energy equivalent in MeV                   2808.391 383          0.000 070             MeV
272: helion mass in u                                       3.014 932 2473        0.000 000 0026        u
273: helion molar mass                                      3.014 932 2473 e-3    0.000 000 0026 e-3    kg mol^-1
274: helion-proton mass ratio                               2.993 152 6713        0.000 000 0026
275: hertz-atomic mass unit relationship                    4.439 821 6294 e-24   0.000 000 0064 e-24   u
276: hertz-electron volt relationship                       4.135 667 33 e-15     0.000 000 10 e-15     eV
277: hertz-hartree relationship                             1.519 829 846 006 e-16 0.000 000 000010e-16 E_h
278: hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
279: hertz-joule relationship                               6.626 068 96 e-34     0.000 000 33 e-34     J
280: hertz-kelvin relationship                              4.799 2374 e-11       0.000 0084 e-11       K
281: hertz-kilogram relationship                            7.372 496 00 e-51     0.000 000 37 e-51     kg
282: inverse fine-structure constant                        137.035 999 679       0.000 000 094
283: inverse meter-atomic mass unit relationship            1.331 025 0394 e-15   0.000 000 0019 e-15   u
284: inverse meter-electron volt relationship               1.239 841 875 e-6     0.000 000 031 e-6     eV
285: inverse meter-hartree relationship                     4.556 335 252 760 e-8 0.000 000 000 030 e-8 E_h
286: inverse meter-hertz relationship                       299 792 458           (exact)               Hz
287: inverse meter-joule relationship                       1.986 445 501 e-25    0.000 000 099 e-25    J
288: inverse meter-kelvin relationship                      1.438 7752 e-2        0.000 0025 e-2        K
289: inverse meter-kilogram relationship                    2.210 218 70 e-42     0.000 000 11 e-42     kg
290: inverse of conductance quantum                         12 906.403 7787       0.000 0088            ohm
291: Josephson constant                                     483 597.891 e9        0.012 e9              Hz V^-1
292: joule-atomic mass unit relationship                    6.700 536 41 e9       0.000 000 33 e9       u
293: joule-electron volt relationship                       6.241 509 65 e18      0.000 000 16 e18      eV
294: joule-hartree relationship                             2.293 712 69 e17      0.000 000 11 e17      E_h
295: joule-hertz relationship                               1.509 190 450 e33     0.000 000 075 e33     Hz
296: joule-inverse meter relationship                       5.034 117 47 e24      0.000 000 25 e24      m^-1
297: joule-kelvin relationship                              7.242 963 e22         0.000 013 e22         K
298: joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
299: kelvin-atomic mass unit relationship                   9.251 098 e-14        0.000 016 e-14        u
300: kelvin-electron volt relationship                      8.617 343 e-5         0.000 015 e-5         eV
301: kelvin-hartree relationship                            3.166 8153 e-6        0.000 0055 e-6        E_h
302: kelvin-hertz relationship                              2.083 6644 e10        0.000 0036 e10        Hz
303: kelvin-inverse meter relationship                      69.503 56             0.000 12              m^-1
304: kelvin-joule relationship                              1.380 6504 e-23       0.000 0024 e-23       J
305: kelvin-kilogram relationship                           1.536 1807 e-40       0.000 0027 e-40       kg
306: kilogram-atomic mass unit relationship                 6.022 141 79 e26      0.000 000 30 e26      u
307: kilogram-electron volt relationship                    5.609 589 12 e35      0.000 000 14 e35      eV
308: kilogram-hartree relationship                          2.061 486 16 e34      0.000 000 10 e34      E_h
309: kilogram-hertz relationship                            1.356 392 733 e50     0.000 000 068 e50     Hz
310: kilogram-inverse meter relationship                    4.524 439 15 e41      0.000 000 23 e41      m^-1
311: kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
312: kilogram-kelvin relationship                           6.509 651 e39         0.000 011 e39         K
313: lattice parameter of silicon                           543.102 064 e-12      0.000 014 e-12        m
314: Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7774 e25        0.000 0047 e25        m^-3
315: mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
316: mag. flux quantum                                      2.067 833 667 e-15    0.000 000 052 e-15    Wb
317: molar gas constant                                     8.314 472             0.000 015             J mol^-1 K^-1
318: molar mass constant                                    1 e-3                 (exact)               kg mol^-1
319: molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
320: molar Planck constant                                  3.990 312 6821 e-10   0.000 000 0057 e-10   J s mol^-1
321: molar Planck constant times c                          0.119 626 564 72      0.000 000 000 17      J m mol^-1
322: molar volume of ideal gas (273.15 K, 100 kPa)          22.710 981 e-3        0.000 040 e-3         m^3 mol^-1
323: molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 996 e-3        0.000 039 e-3         m^3 mol^-1
324: molar volume of silicon                                12.058 8349 e-6       0.000 0011 e-6        m^3 mol^-1
325: Mo x unit                                              1.002 099 55 e-13     0.000 000 53 e-13     m
326: muon Compton wavelength                                11.734 441 04 e-15    0.000 000 30 e-15     m
327: muon Compton wavelength over 2 pi                      1.867 594 295 e-15    0.000 000 047 e-15    m
328: muon-electron mass ratio                               206.768 2823          0.000 0052
329: muon g factor                                          -2.002 331 8414       0.000 000 0012
330: muon mag. mom.                                         -4.490 447 86 e-26    0.000 000 16 e-26     J T^-1
331: muon mag. mom. anomaly                                 1.165 920 69 e-3      0.000 000 60 e-3
332: muon mag. mom. to Bohr magneton ratio                  -4.841 970 49 e-3     0.000 000 12 e-3
333: muon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 23
334: muon mass                                              1.883 531 30 e-28     0.000 000 11 e-28     kg
335: muon mass energy equivalent                            1.692 833 510 e-11    0.000 000 095 e-11    J
336: muon mass energy equivalent in MeV                     105.658 3668          0.000 0038            MeV
337: muon mass in u                                         0.113 428 9256        0.000 000 0029        u
338: muon molar mass                                        0.113 428 9256 e-3    0.000 000 0029 e-3    kg mol^-1
339: muon-neutron mass ratio                                0.112 454 5167        0.000 000 0029
340: muon-proton mag. mom. ratio                            -3.183 345 137        0.000 000 085
341: muon-proton mass ratio                                 0.112 609 5261        0.000 000 0029
342: muon-tau mass ratio                                    5.945 92 e-2          0.000 97 e-2
343: natural unit of action                                 1.054 571 628 e-34    0.000 000 053 e-34    J s
344: natural unit of action in eV s                         6.582 118 99 e-16     0.000 000 16 e-16     eV s
345: natural unit of energy                                 8.187 104 38 e-14     0.000 000 41 e-14     J
346: natural unit of energy in MeV                          0.510 998 910         0.000 000 013         MeV
347: natural unit of length                                 386.159 264 59 e-15   0.000 000 53 e-15     m
348: natural unit of mass                                   9.109 382 15 e-31     0.000 000 45 e-31     kg
349: natural unit of momentum                               2.730 924 06 e-22     0.000 000 14 e-22     kg m s^-1
350: natural unit of momentum in MeV/c                      0.510 998 910         0.000 000 013         MeV/c
351: natural unit of time                                   1.288 088 6570 e-21   0.000 000 0018 e-21   s
352: natural unit of velocity                               299 792 458           (exact)               m s^-1
353: neutron Compton wavelength                             1.319 590 8951 e-15   0.000 000 0020 e-15   m
354: neutron Compton wavelength over 2 pi                   0.210 019 413 82 e-15 0.000 000 000 31 e-15 m
355: neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
356: neutron-electron mass ratio                            1838.683 6605         0.000 0011
357: neutron g factor                                       -3.826 085 45         0.000 000 90
358: neutron gyromag. ratio                                 1.832 471 85 e8       0.000 000 43 e8       s^-1 T^-1
359: neutron gyromag. ratio over 2 pi                       29.164 6954           0.000 0069            MHz T^-1
360: neutron mag. mom.                                      -0.966 236 41 e-26    0.000 000 23 e-26     J T^-1
361: neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
362: neutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45
363: neutron mass                                           1.674 927 211 e-27    0.000 000 084 e-27    kg
364: neutron mass energy equivalent                         1.505 349 505 e-10    0.000 000 075 e-10    J
365: neutron mass energy equivalent in MeV                  939.565 346           0.000 023             MeV
366: neutron mass in u                                      1.008 664 915 97      0.000 000 000 43      u
367: neutron molar mass                                     1.008 664 915 97 e-3  0.000 000 000 43 e-3  kg mol^-1
368: neutron-muon mass ratio                                8.892 484 09          0.000 000 23
369: neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
370: neutron-proton mass ratio                              1.001 378 419 18      0.000 000 000 46
371: neutron-tau mass ratio                                 0.528 740             0.000 086
372: neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
373: Newtonian constant of gravitation                      6.674 28 e-11         0.000 67 e-11         m^3 kg^-1 s^-2
374: Newtonian constant of gravitation over h-bar c         6.708 81 e-39         0.000 67 e-39         (GeV/c^2)^-2
375: nuclear magneton                                       5.050 783 24 e-27     0.000 000 13 e-27     J T^-1
376: nuclear magneton in eV/T                               3.152 451 2326 e-8    0.000 000 0045 e-8    eV T^-1
377: nuclear magneton in inverse meters per tesla           2.542 623 616 e-2     0.000 000 064 e-2     m^-1 T^-1
378: nuclear magneton in K/T                                3.658 2637 e-4        0.000 0064 e-4        K T^-1
379: nuclear magneton in MHz/T                              7.622 593 84          0.000 000 19          MHz T^-1
380: Planck constant                                        6.626 068 96 e-34     0.000 000 33 e-34     J s
381: Planck constant in eV s                                4.135 667 33 e-15     0.000 000 10 e-15     eV s
382: Planck constant over 2 pi                              1.054 571 628 e-34    0.000 000 053 e-34    J s
383: Planck constant over 2 pi in eV s                      6.582 118 99 e-16     0.000 000 16 e-16     eV s
384: Planck constant over 2 pi times c in MeV fm            197.326 9631          0.000 0049            MeV fm
385: Planck length                                          1.616 252 e-35        0.000 081 e-35        m
386: Planck mass                                            2.176 44 e-8          0.000 11 e-8          kg
387: Planck mass energy equivalent in GeV                   1.220 892 e19         0.000 061 e19         GeV
388: Planck temperature                                     1.416 785 e32         0.000 071 e32         K
389: Planck time                                            5.391 24 e-44         0.000 27 e-44         s
390: proton charge to mass quotient                         9.578 833 92 e7       0.000 000 24 e7       C kg^-1
391: proton Compton wavelength                              1.321 409 8446 e-15   0.000 000 0019 e-15   m
392: proton Compton wavelength over 2 pi                    0.210 308 908 61 e-15 0.000 000 000 30 e-15 m
393: proton-electron mass ratio                             1836.152 672 47       0.000 000 80
394: proton g factor                                        5.585 694 713         0.000 000 046
395: proton gyromag. ratio                                  2.675 222 099 e8      0.000 000 070 e8      s^-1 T^-1
396: proton gyromag. ratio over 2 pi                        42.577 4821           0.000 0011            MHz T^-1
397: proton mag. mom.                                       1.410 606 662 e-26    0.000 000 037 e-26    J T^-1
398: proton mag. mom. to Bohr magneton ratio                1.521 032 209 e-3     0.000 000 012 e-3
399: proton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023
400: proton mag. shielding correction                       25.694 e-6            0.014 e-6
401: proton mass                                            1.672 621 637 e-27    0.000 000 083 e-27    kg
402: proton mass energy equivalent                          1.503 277 359 e-10    0.000 000 075 e-10    J
403: proton mass energy equivalent in MeV                   938.272 013           0.000 023             MeV
404: proton mass in u                                       1.007 276 466 77      0.000 000 000 10      u
405: proton molar mass                                      1.007 276 466 77 e-3  0.000 000 000 10 e-3  kg mol^-1
406: proton-muon mass ratio                                 8.880 243 39          0.000 000 23
407: proton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34
408: proton-neutron mass ratio                              0.998 623 478 24      0.000 000 000 46
409: proton rms charge radius                               0.8768 e-15           0.0069 e-15           m
410: proton-tau mass ratio                                  0.528 012             0.000 086
411: quantum of circulation                                 3.636 947 5199 e-4    0.000 000 0050 e-4    m^2 s^-1
412: quantum of circulation times 2                         7.273 895 040 e-4     0.000 000 010 e-4     m^2 s^-1
413: Rydberg constant                                       10 973 731.568 527    0.000 073             m^-1
414: Rydberg constant times c in Hz                         3.289 841 960 361 e15 0.000 000 000 022 e15 Hz
415: Rydberg constant times hc in eV                        13.605 691 93         0.000 000 34          eV
416: Rydberg constant times hc in J                         2.179 871 97 e-18     0.000 000 11 e-18     J
417: Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7047           0.000 0044
418: Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8677           0.000 0044
419: second radiation constant                              1.438 7752 e-2        0.000 0025 e-2        m K
420: shielded helion gyromag. ratio                         2.037 894 730 e8      0.000 000 056 e8      s^-1 T^-1
421: shielded helion gyromag. ratio over 2 pi               32.434 101 98         0.000 000 90          MHz T^-1
422: shielded helion mag. mom.                              -1.074 552 982 e-26   0.000 000 030 e-26    J T^-1
423: shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
424: shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025
425: shielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011
426: shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
427: shielded proton gyromag. ratio                         2.675 153 362 e8      0.000 000 073 e8      s^-1 T^-1
428: shielded proton gyromag. ratio over 2 pi               42.576 3881           0.000 0012            MHz T^-1
429: shielded proton mag. mom.                              1.410 570 419 e-26    0.000 000 038 e-26    J T^-1
430: shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
431: shielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030
432: speed of light in vacuum                               299 792 458           (exact)               m s^-1
433: standard acceleration of gravity                       9.806 65              (exact)               m s^-2
434: standard atmosphere                                    101 325               (exact)               Pa
435: Stefan-Boltzmann constant                              5.670 400 e-8         0.000 040 e-8         W m^-2 K^-4
436: tau Compton wavelength                                 0.697 72 e-15         0.000 11 e-15         m
437: tau Compton wavelength over 2 pi                       0.111 046 e-15        0.000 018 e-15        m
438: tau-electron mass ratio                                3477.48               0.57
439: tau mass                                               3.167 77 e-27         0.000 52 e-27         kg
440: tau mass energy equivalent                             2.847 05 e-10         0.000 46 e-10         J
441: tau mass energy equivalent in MeV                      1776.99               0.29                  MeV
442: tau mass in u                                          1.907 68              0.000 31              u
443: tau molar mass                                         1.907 68 e-3          0.000 31 e-3          kg mol^-1
444: tau-muon mass ratio                                    16.8183               0.0027
445: tau-neutron mass ratio                                 1.891 29              0.000 31
446: tau-proton mass ratio                                  1.893 90              0.000 31
447: Thomson cross section                                  0.665 245 8558 e-28   0.000 000 0027 e-28   m^2
448: triton-electron mag. mom. ratio                        -1.620 514 423 e-3    0.000 000 021 e-3
449: triton-electron mass ratio                             5496.921 5269         0.000 0051
450: triton g factor                                        5.957 924 896         0.000 000 076
451: triton mag. mom.                                       1.504 609 361 e-26    0.000 000 042 e-26    J T^-1
452: triton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3
453: triton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038
454: triton mass                                            5.007 355 88 e-27     0.000 000 25 e-27     kg
455: triton mass energy equivalent                          4.500 387 03 e-10     0.000 000 22 e-10     J
456: triton mass energy equivalent in MeV                   2808.920 906          0.000 070             MeV
457: triton mass in u                                       3.015 500 7134        0.000 000 0025        u
458: triton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1
459: triton-neutron mag. mom. ratio                         -1.557 185 53         0.000 000 37
460: triton-proton mag. mom. ratio                          1.066 639 908         0.000 000 010
461: triton-proton mass ratio                               2.993 717 0309        0.000 000 0025
462: unified atomic mass unit                               1.660 538 782 e-27    0.000 000 083 e-27    kg
463: von Klitzing constant                                  25 812.807 557        0.000 018             ohm
464: weak mixing angle                                      0.222 55              0.000 56
465: Wien frequency displacement law constant               5.878 933 e10         0.000 010 e10         Hz K^-1
466: Wien wavelength displacement law constant              2.897 7685 e-3        0.000 0051 e-3        m K'''
467: 
468: txt2010 = '''\
469: {220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m
470: alpha particle-electron mass ratio                     7294.299 5361         0.000 0029
471: alpha particle mass                                    6.644 656 75 e-27     0.000 000 29 e-27     kg
472: alpha particle mass energy equivalent                  5.971 919 67 e-10     0.000 000 26 e-10     J
473: alpha particle mass energy equivalent in MeV           3727.379 240          0.000 082             MeV
474: alpha particle mass in u                               4.001 506 179 125     0.000 000 000 062     u
475: alpha particle molar mass                              4.001 506 179 125 e-3 0.000 000 000 062 e-3 kg mol^-1
476: alpha particle-proton mass ratio                       3.972 599 689 33      0.000 000 000 36
477: Angstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m
478: atomic mass constant                                   1.660 538 921 e-27    0.000 000 073 e-27    kg
479: atomic mass constant energy equivalent                 1.492 417 954 e-10    0.000 000 066 e-10    J
480: atomic mass constant energy equivalent in MeV          931.494 061           0.000 021             MeV
481: atomic mass unit-electron volt relationship            931.494 061 e6        0.000 021 e6          eV
482: atomic mass unit-hartree relationship                  3.423 177 6845 e7     0.000 000 0024 e7     E_h
483: atomic mass unit-hertz relationship                    2.252 342 7168 e23    0.000 000 0016 e23    Hz
484: atomic mass unit-inverse meter relationship            7.513 006 6042 e14    0.000 000 0053 e14    m^-1
485: atomic mass unit-joule relationship                    1.492 417 954 e-10    0.000 000 066 e-10    J
486: atomic mass unit-kelvin relationship                   1.080 954 08 e13      0.000 000 98 e13      K
487: atomic mass unit-kilogram relationship                 1.660 538 921 e-27    0.000 000 073 e-27    kg
488: atomic unit of 1st hyperpolarizability                 3.206 361 449 e-53    0.000 000 071 e-53    C^3 m^3 J^-2
489: atomic unit of 2nd hyperpolarizability                 6.235 380 54 e-65     0.000 000 28 e-65     C^4 m^4 J^-3
490: atomic unit of action                                  1.054 571 726 e-34    0.000 000 047 e-34    J s
491: atomic unit of charge                                  1.602 176 565 e-19    0.000 000 035 e-19    C
492: atomic unit of charge density                          1.081 202 338 e12     0.000 000 024 e12     C m^-3
493: atomic unit of current                                 6.623 617 95 e-3      0.000 000 15 e-3      A
494: atomic unit of electric dipole mom.                    8.478 353 26 e-30     0.000 000 19 e-30     C m
495: atomic unit of electric field                          5.142 206 52 e11      0.000 000 11 e11      V m^-1
496: atomic unit of electric field gradient                 9.717 362 00 e21      0.000 000 21 e21      V m^-2
497: atomic unit of electric polarizability                 1.648 777 2754 e-41   0.000 000 0016 e-41   C^2 m^2 J^-1
498: atomic unit of electric potential                      27.211 385 05         0.000 000 60          V
499: atomic unit of electric quadrupole mom.                4.486 551 331 e-40    0.000 000 099 e-40    C m^2
500: atomic unit of energy                                  4.359 744 34 e-18     0.000 000 19 e-18     J
501: atomic unit of force                                   8.238 722 78 e-8      0.000 000 36 e-8      N
502: atomic unit of length                                  0.529 177 210 92 e-10 0.000 000 000 17 e-10 m
503: atomic unit of mag. dipole mom.                        1.854 801 936 e-23    0.000 000 041 e-23    J T^-1
504: atomic unit of mag. flux density                       2.350 517 464 e5      0.000 000 052 e5      T
505: atomic unit of magnetizability                         7.891 036 607 e-29    0.000 000 013 e-29    J T^-2
506: atomic unit of mass                                    9.109 382 91 e-31     0.000 000 40 e-31     kg
507: atomic unit of mom.um                                  1.992 851 740 e-24    0.000 000 088 e-24    kg m s^-1
508: atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
509: atomic unit of time                                    2.418 884 326 502e-17 0.000 000 000 012e-17 s
510: atomic unit of velocity                                2.187 691 263 79 e6   0.000 000 000 71 e6   m s^-1
511: Avogadro constant                                      6.022 141 29 e23      0.000 000 27 e23      mol^-1
512: Bohr magneton                                          927.400 968 e-26      0.000 020 e-26        J T^-1
513: Bohr magneton in eV/T                                  5.788 381 8066 e-5    0.000 000 0038 e-5    eV T^-1
514: Bohr magneton in Hz/T                                  13.996 245 55 e9      0.000 000 31 e9       Hz T^-1
515: Bohr magneton in inverse meters per tesla              46.686 4498           0.000 0010            m^-1 T^-1
516: Bohr magneton in K/T                                   0.671 713 88          0.000 000 61          K T^-1
517: Bohr radius                                            0.529 177 210 92 e-10 0.000 000 000 17 e-10 m
518: Boltzmann constant                                     1.380 6488 e-23       0.000 0013 e-23       J K^-1
519: Boltzmann constant in eV/K                             8.617 3324 e-5        0.000 0078 e-5        eV K^-1
520: Boltzmann constant in Hz/K                             2.083 6618 e10        0.000 0019 e10        Hz K^-1
521: Boltzmann constant in inverse meters per kelvin        69.503 476            0.000 063             m^-1 K^-1
522: characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
523: classical electron radius                              2.817 940 3267 e-15   0.000 000 0027 e-15   m
524: Compton wavelength                                     2.426 310 2389 e-12   0.000 000 0016 e-12   m
525: Compton wavelength over 2 pi                           386.159 268 00 e-15   0.000 000 25 e-15     m
526: conductance quantum                                    7.748 091 7346 e-5    0.000 000 0025 e-5    S
527: conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
528: conventional value of von Klitzing constant            25 812.807            (exact)               ohm
529: Cu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m
530: deuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4
531: deuteron-electron mass ratio                           3670.482 9652         0.000 0015
532: deuteron g factor                                      0.857 438 2308        0.000 000 0072
533: deuteron mag. mom.                                     0.433 073 489 e-26    0.000 000 010 e-26    J T^-1
534: deuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3
535: deuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072
536: deuteron mass                                          3.343 583 48 e-27     0.000 000 15 e-27     kg
537: deuteron mass energy equivalent                        3.005 062 97 e-10     0.000 000 13 e-10     J
538: deuteron mass energy equivalent in MeV                 1875.612 859          0.000 041             MeV
539: deuteron mass in u                                     2.013 553 212 712     0.000 000 000 077     u
540: deuteron molar mass                                    2.013 553 212 712 e-3 0.000 000 000 077 e-3 kg mol^-1
541: deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
542: deuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024
543: deuteron-proton mass ratio                             1.999 007 500 97      0.000 000 000 18
544: deuteron rms charge radius                             2.1424 e-15           0.0021 e-15           m
545: electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
546: electron charge to mass quotient                       -1.758 820 088 e11    0.000 000 039 e11     C kg^-1
547: electron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018
548: electron-deuteron mass ratio                           2.724 437 1095 e-4    0.000 000 0011 e-4
549: electron g factor                                      -2.002 319 304 361 53 0.000 000 000 000 53
550: electron gyromag. ratio                                1.760 859 708 e11     0.000 000 039 e11     s^-1 T^-1
551: electron gyromag. ratio over 2 pi                      28 024.952 66         0.000 62              MHz T^-1
552: electron-helion mass ratio                             1.819 543 0761 e-4    0.000 000 0017 e-4
553: electron mag. mom.                                     -928.476 430 e-26     0.000 021 e-26        J T^-1
554: electron mag. mom. anomaly                             1.159 652 180 76 e-3  0.000 000 000 27 e-3
555: electron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 76 0.000 000 000 000 27
556: electron mag. mom. to nuclear magneton ratio           -1838.281 970 90      0.000 000 75
557: electron mass                                          9.109 382 91 e-31     0.000 000 40 e-31     kg
558: electron mass energy equivalent                        8.187 105 06 e-14     0.000 000 36 e-14     J
559: electron mass energy equivalent in MeV                 0.510 998 928         0.000 000 011         MeV
560: electron mass in u                                     5.485 799 0946 e-4    0.000 000 0022 e-4    u
561: electron molar mass                                    5.485 799 0946 e-7    0.000 000 0022 e-7    kg mol^-1
562: electron-muon mag. mom. ratio                          206.766 9896          0.000 0052
563: electron-muon mass ratio                               4.836 331 66 e-3      0.000 000 12 e-3
564: electron-neutron mag. mom. ratio                       960.920 50            0.000 23
565: electron-neutron mass ratio                            5.438 673 4461 e-4    0.000 000 0032 e-4
566: electron-proton mag. mom. ratio                        -658.210 6848         0.000 0054
567: electron-proton mass ratio                             5.446 170 2178 e-4    0.000 000 0022 e-4
568: electron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4
569: electron to alpha particle mass ratio                  1.370 933 555 78 e-4  0.000 000 000 55 e-4
570: electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
571: electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
572: electron-triton mass ratio                             1.819 200 0653 e-4    0.000 000 0017 e-4
573: electron volt                                          1.602 176 565 e-19    0.000 000 035 e-19    J
574: electron volt-atomic mass unit relationship            1.073 544 150 e-9     0.000 000 024 e-9     u
575: electron volt-hartree relationship                     3.674 932 379 e-2     0.000 000 081 e-2     E_h
576: electron volt-hertz relationship                       2.417 989 348 e14     0.000 000 053 e14     Hz
577: electron volt-inverse meter relationship               8.065 544 29 e5       0.000 000 18 e5       m^-1
578: electron volt-joule relationship                       1.602 176 565 e-19    0.000 000 035 e-19    J
579: electron volt-kelvin relationship                      1.160 4519 e4         0.000 0011 e4         K
580: electron volt-kilogram relationship                    1.782 661 845 e-36    0.000 000 039 e-36    kg
581: elementary charge                                      1.602 176 565 e-19    0.000 000 035 e-19    C
582: elementary charge over h                               2.417 989 348 e14     0.000 000 053 e14     A J^-1
583: Faraday constant                                       96 485.3365           0.0021                C mol^-1
584: Faraday constant for conventional electric current     96 485.3321           0.0043                C_90 mol^-1
585: Fermi coupling constant                                1.166 364 e-5         0.000 005 e-5         GeV^-2
586: fine-structure constant                                7.297 352 5698 e-3    0.000 000 0024 e-3
587: first radiation constant                               3.741 771 53 e-16     0.000 000 17 e-16     W m^2
588: first radiation constant for spectral radiance         1.191 042 869 e-16    0.000 000 053 e-16    W m^2 sr^-1
589: hartree-atomic mass unit relationship                  2.921 262 3246 e-8    0.000 000 0021 e-8    u
590: hartree-electron volt relationship                     27.211 385 05         0.000 000 60          eV
591: Hartree energy                                         4.359 744 34 e-18     0.000 000 19 e-18     J
592: Hartree energy in eV                                   27.211 385 05         0.000 000 60          eV
593: hartree-hertz relationship                             6.579 683 920 729 e15 0.000 000 000 033 e15 Hz
594: hartree-inverse meter relationship                     2.194 746 313 708 e7  0.000 000 000 011 e7  m^-1
595: hartree-joule relationship                             4.359 744 34 e-18     0.000 000 19 e-18     J
596: hartree-kelvin relationship                            3.157 7504 e5         0.000 0029 e5         K
597: hartree-kilogram relationship                          4.850 869 79 e-35     0.000 000 21 e-35     kg
598: helion-electron mass ratio                             5495.885 2754         0.000 0050
599: helion g factor                                        -4.255 250 613        0.000 000 050
600: helion mag. mom.                                       -1.074 617 486 e-26   0.000 000 027 e-26    J T^-1
601: helion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3
602: helion mag. mom. to nuclear magneton ratio             -2.127 625 306        0.000 000 025
603: helion mass                                            5.006 412 34 e-27     0.000 000 22 e-27     kg
604: helion mass energy equivalent                          4.499 539 02 e-10     0.000 000 20 e-10     J
605: helion mass energy equivalent in MeV                   2808.391 482          0.000 062             MeV
606: helion mass in u                                       3.014 932 2468        0.000 000 0025        u
607: helion molar mass                                      3.014 932 2468 e-3    0.000 000 0025 e-3    kg mol^-1
608: helion-proton mass ratio                               2.993 152 6707        0.000 000 0025
609: hertz-atomic mass unit relationship                    4.439 821 6689 e-24   0.000 000 0031 e-24   u
610: hertz-electron volt relationship                       4.135 667 516 e-15    0.000 000 091 e-15    eV
611: hertz-hartree relationship                             1.519 829 8460045e-16 0.000 000 0000076e-16 E_h
612: hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
613: hertz-joule relationship                               6.626 069 57 e-34     0.000 000 29 e-34     J
614: hertz-kelvin relationship                              4.799 2434 e-11       0.000 0044 e-11       K
615: hertz-kilogram relationship                            7.372 496 68 e-51     0.000 000 33 e-51     kg
616: inverse fine-structure constant                        137.035 999 074       0.000 000 044
617: inverse meter-atomic mass unit relationship            1.331 025 051 20 e-15 0.000 000 000 94 e-15 u
618: inverse meter-electron volt relationship               1.239 841 930 e-6     0.000 000 027 e-6     eV
619: inverse meter-hartree relationship                     4.556 335 252 755 e-8 0.000 000 000 023 e-8 E_h
620: inverse meter-hertz relationship                       299 792 458           (exact)               Hz
621: inverse meter-joule relationship                       1.986 445 684 e-25    0.000 000 088 e-25    J
622: inverse meter-kelvin relationship                      1.438 7770 e-2        0.000 0013 e-2        K
623: inverse meter-kilogram relationship                    2.210 218 902 e-42    0.000 000 098 e-42    kg
624: inverse of conductance quantum                         12 906.403 7217       0.000 0042            ohm
625: Josephson constant                                     483 597.870 e9        0.011 e9              Hz V^-1
626: joule-atomic mass unit relationship                    6.700 535 85 e9       0.000 000 30 e9       u
627: joule-electron volt relationship                       6.241 509 34 e18      0.000 000 14 e18      eV
628: joule-hartree relationship                             2.293 712 48 e17      0.000 000 10 e17      E_h
629: joule-hertz relationship                               1.509 190 311 e33     0.000 000 067 e33     Hz
630: joule-inverse meter relationship                       5.034 117 01 e24      0.000 000 22 e24      m^-1
631: joule-kelvin relationship                              7.242 9716 e22        0.000 0066 e22        K
632: joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
633: kelvin-atomic mass unit relationship                   9.251 0868 e-14       0.000 0084 e-14       u
634: kelvin-electron volt relationship                      8.617 3324 e-5        0.000 0078 e-5        eV
635: kelvin-hartree relationship                            3.166 8114 e-6        0.000 0029 e-6        E_h
636: kelvin-hertz relationship                              2.083 6618 e10        0.000 0019 e10        Hz
637: kelvin-inverse meter relationship                      69.503 476            0.000 063             m^-1
638: kelvin-joule relationship                              1.380 6488 e-23       0.000 0013 e-23       J
639: kelvin-kilogram relationship                           1.536 1790 e-40       0.000 0014 e-40       kg
640: kilogram-atomic mass unit relationship                 6.022 141 29 e26      0.000 000 27 e26      u
641: kilogram-electron volt relationship                    5.609 588 85 e35      0.000 000 12 e35      eV
642: kilogram-hartree relationship                          2.061 485 968 e34     0.000 000 091 e34     E_h
643: kilogram-hertz relationship                            1.356 392 608 e50     0.000 000 060 e50     Hz
644: kilogram-inverse meter relationship                    4.524 438 73 e41      0.000 000 20 e41      m^-1
645: kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
646: kilogram-kelvin relationship                           6.509 6582 e39        0.000 0059 e39        K
647: lattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m
648: Loschmidt constant (273.15 K, 100 kPa)                 2.651 6462 e25        0.000 0024 e25        m^-3
649: Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7805 e25        0.000 0024 e25        m^-3
650: mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
651: mag. flux quantum                                      2.067 833 758 e-15    0.000 000 046 e-15    Wb
652: molar gas constant                                     8.314 4621            0.000 0075            J mol^-1 K^-1
653: molar mass constant                                    1 e-3                 (exact)               kg mol^-1
654: molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
655: molar Planck constant                                  3.990 312 7176 e-10   0.000 000 0028 e-10   J s mol^-1
656: molar Planck constant times c                          0.119 626 565 779     0.000 000 000 084     J m mol^-1
657: molar volume of ideal gas (273.15 K, 100 kPa)          22.710 953 e-3        0.000 021 e-3         m^3 mol^-1
658: molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 968 e-3        0.000 020 e-3         m^3 mol^-1
659: molar volume of silicon                                12.058 833 01 e-6     0.000 000 80 e-6      m^3 mol^-1
660: Mo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m
661: muon Compton wavelength                                11.734 441 03 e-15    0.000 000 30 e-15     m
662: muon Compton wavelength over 2 pi                      1.867 594 294 e-15    0.000 000 047 e-15    m
663: muon-electron mass ratio                               206.768 2843          0.000 0052
664: muon g factor                                          -2.002 331 8418       0.000 000 0013
665: muon mag. mom.                                         -4.490 448 07 e-26    0.000 000 15 e-26     J T^-1
666: muon mag. mom. anomaly                                 1.165 920 91 e-3      0.000 000 63 e-3
667: muon mag. mom. to Bohr magneton ratio                  -4.841 970 44 e-3     0.000 000 12 e-3
668: muon mag. mom. to nuclear magneton ratio               -8.890 596 97         0.000 000 22
669: muon mass                                              1.883 531 475 e-28    0.000 000 096 e-28    kg
670: muon mass energy equivalent                            1.692 833 667 e-11    0.000 000 086 e-11    J
671: muon mass energy equivalent in MeV                     105.658 3715          0.000 0035            MeV
672: muon mass in u                                         0.113 428 9267        0.000 000 0029        u
673: muon molar mass                                        0.113 428 9267 e-3    0.000 000 0029 e-3    kg mol^-1
674: muon-neutron mass ratio                                0.112 454 5177        0.000 000 0028
675: muon-proton mag. mom. ratio                            -3.183 345 107        0.000 000 084
676: muon-proton mass ratio                                 0.112 609 5272        0.000 000 0028
677: muon-tau mass ratio                                    5.946 49 e-2          0.000 54 e-2
678: natural unit of action                                 1.054 571 726 e-34    0.000 000 047 e-34    J s
679: natural unit of action in eV s                         6.582 119 28 e-16     0.000 000 15 e-16     eV s
680: natural unit of energy                                 8.187 105 06 e-14     0.000 000 36 e-14     J
681: natural unit of energy in MeV                          0.510 998 928         0.000 000 011         MeV
682: natural unit of length                                 386.159 268 00 e-15   0.000 000 25 e-15     m
683: natural unit of mass                                   9.109 382 91 e-31     0.000 000 40 e-31     kg
684: natural unit of mom.um                                 2.730 924 29 e-22     0.000 000 12 e-22     kg m s^-1
685: natural unit of mom.um in MeV/c                        0.510 998 928         0.000 000 011         MeV/c
686: natural unit of time                                   1.288 088 668 33 e-21 0.000 000 000 83 e-21 s
687: natural unit of velocity                               299 792 458           (exact)               m s^-1
688: neutron Compton wavelength                             1.319 590 9068 e-15   0.000 000 0011 e-15   m
689: neutron Compton wavelength over 2 pi                   0.210 019 415 68 e-15 0.000 000 000 17 e-15 m
690: neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
691: neutron-electron mass ratio                            1838.683 6605         0.000 0011
692: neutron g factor                                       -3.826 085 45         0.000 000 90
693: neutron gyromag. ratio                                 1.832 471 79 e8       0.000 000 43 e8       s^-1 T^-1
694: neutron gyromag. ratio over 2 pi                       29.164 6943           0.000 0069            MHz T^-1
695: neutron mag. mom.                                      -0.966 236 47 e-26    0.000 000 23 e-26     J T^-1
696: neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
697: neutron mag. mom. to nuclear magneton ratio            -1.913 042 72         0.000 000 45
698: neutron mass                                           1.674 927 351 e-27    0.000 000 074 e-27    kg
699: neutron mass energy equivalent                         1.505 349 631 e-10    0.000 000 066 e-10    J
700: neutron mass energy equivalent in MeV                  939.565 379           0.000 021             MeV
701: neutron mass in u                                      1.008 664 916 00      0.000 000 000 43      u
702: neutron molar mass                                     1.008 664 916 00 e-3  0.000 000 000 43 e-3  kg mol^-1
703: neutron-muon mass ratio                                8.892 484 00          0.000 000 22
704: neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
705: neutron-proton mass difference                         2.305 573 92 e-30     0.000 000 76 e-30
706: neutron-proton mass difference energy equivalent       2.072 146 50 e-13     0.000 000 68 e-13
707: neutron-proton mass difference energy equivalent in MeV 1.293 332 17          0.000 000 42
708: neutron-proton mass difference in u                    0.001 388 449 19      0.000 000 000 45
709: neutron-proton mass ratio                              1.001 378 419 17      0.000 000 000 45
710: neutron-tau mass ratio                                 0.528 790             0.000 048
711: neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
712: Newtonian constant of gravitation                      6.673 84 e-11         0.000 80 e-11         m^3 kg^-1 s^-2
713: Newtonian constant of gravitation over h-bar c         6.708 37 e-39         0.000 80 e-39         (GeV/c^2)^-2
714: nuclear magneton                                       5.050 783 53 e-27     0.000 000 11 e-27     J T^-1
715: nuclear magneton in eV/T                               3.152 451 2605 e-8    0.000 000 0022 e-8    eV T^-1
716: nuclear magneton in inverse meters per tesla           2.542 623 527 e-2     0.000 000 056 e-2     m^-1 T^-1
717: nuclear magneton in K/T                                3.658 2682 e-4        0.000 0033 e-4        K T^-1
718: nuclear magneton in MHz/T                              7.622 593 57          0.000 000 17          MHz T^-1
719: Planck constant                                        6.626 069 57 e-34     0.000 000 29 e-34     J s
720: Planck constant in eV s                                4.135 667 516 e-15    0.000 000 091 e-15    eV s
721: Planck constant over 2 pi                              1.054 571 726 e-34    0.000 000 047 e-34    J s
722: Planck constant over 2 pi in eV s                      6.582 119 28 e-16     0.000 000 15 e-16     eV s
723: Planck constant over 2 pi times c in MeV fm            197.326 9718          0.000 0044            MeV fm
724: Planck length                                          1.616 199 e-35        0.000 097 e-35        m
725: Planck mass                                            2.176 51 e-8          0.000 13 e-8          kg
726: Planck mass energy equivalent in GeV                   1.220 932 e19         0.000 073 e19         GeV
727: Planck temperature                                     1.416 833 e32         0.000 085 e32         K
728: Planck time                                            5.391 06 e-44         0.000 32 e-44         s
729: proton charge to mass quotient                         9.578 833 58 e7       0.000 000 21 e7       C kg^-1
730: proton Compton wavelength                              1.321 409 856 23 e-15 0.000 000 000 94 e-15 m
731: proton Compton wavelength over 2 pi                    0.210 308 910 47 e-15 0.000 000 000 15 e-15 m
732: proton-electron mass ratio                             1836.152 672 45       0.000 000 75
733: proton g factor                                        5.585 694 713         0.000 000 046
734: proton gyromag. ratio                                  2.675 222 005 e8      0.000 000 063 e8      s^-1 T^-1
735: proton gyromag. ratio over 2 pi                        42.577 4806           0.000 0010            MHz T^-1
736: proton mag. mom.                                       1.410 606 743 e-26    0.000 000 033 e-26    J T^-1
737: proton mag. mom. to Bohr magneton ratio                1.521 032 210 e-3     0.000 000 012 e-3
738: proton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023
739: proton mag. shielding correction                       25.694 e-6            0.014 e-6
740: proton mass                                            1.672 621 777 e-27    0.000 000 074 e-27    kg
741: proton mass energy equivalent                          1.503 277 484 e-10    0.000 000 066 e-10    J
742: proton mass energy equivalent in MeV                   938.272 046           0.000 021             MeV
743: proton mass in u                                       1.007 276 466 812     0.000 000 000 090     u
744: proton molar mass                                      1.007 276 466 812 e-3 0.000 000 000 090 e-3 kg mol^-1
745: proton-muon mass ratio                                 8.880 243 31          0.000 000 22
746: proton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34
747: proton-neutron mass ratio                              0.998 623 478 26      0.000 000 000 45
748: proton rms charge radius                               0.8775 e-15           0.0051 e-15           m
749: proton-tau mass ratio                                  0.528 063             0.000 048
750: quantum of circulation                                 3.636 947 5520 e-4    0.000 000 0024 e-4    m^2 s^-1
751: quantum of circulation times 2                         7.273 895 1040 e-4    0.000 000 0047 e-4    m^2 s^-1
752: Rydberg constant                                       10 973 731.568 539    0.000 055             m^-1
753: Rydberg constant times c in Hz                         3.289 841 960 364 e15 0.000 000 000 017 e15 Hz
754: Rydberg constant times hc in eV                        13.605 692 53         0.000 000 30          eV
755: Rydberg constant times hc in J                         2.179 872 171 e-18    0.000 000 096 e-18    J
756: Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7078           0.000 0023
757: Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8708           0.000 0023
758: second radiation constant                              1.438 7770 e-2        0.000 0013 e-2        m K
759: shielded helion gyromag. ratio                         2.037 894 659 e8      0.000 000 051 e8      s^-1 T^-1
760: shielded helion gyromag. ratio over 2 pi               32.434 100 84         0.000 000 81          MHz T^-1
761: shielded helion mag. mom.                              -1.074 553 044 e-26   0.000 000 027 e-26    J T^-1
762: shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
763: shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025
764: shielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011
765: shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
766: shielded proton gyromag. ratio                         2.675 153 268 e8      0.000 000 066 e8      s^-1 T^-1
767: shielded proton gyromag. ratio over 2 pi               42.576 3866           0.000 0010            MHz T^-1
768: shielded proton mag. mom.                              1.410 570 499 e-26    0.000 000 035 e-26    J T^-1
769: shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
770: shielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030
771: speed of light in vacuum                               299 792 458           (exact)               m s^-1
772: standard acceleration of gravity                       9.806 65              (exact)               m s^-2
773: standard atmosphere                                    101 325               (exact)               Pa
774: standard-state pressure                                100 000               (exact)               Pa
775: Stefan-Boltzmann constant                              5.670 373 e-8         0.000 021 e-8         W m^-2 K^-4
776: tau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m
777: tau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m
778: tau-electron mass ratio                                3477.15               0.31
779: tau mass                                               3.167 47 e-27         0.000 29 e-27         kg
780: tau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J
781: tau mass energy equivalent in MeV                      1776.82               0.16                  MeV
782: tau mass in u                                          1.907 49              0.000 17              u
783: tau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1
784: tau-muon mass ratio                                    16.8167               0.0015
785: tau-neutron mass ratio                                 1.891 11              0.000 17
786: tau-proton mass ratio                                  1.893 72              0.000 17
787: Thomson cross section                                  0.665 245 8734 e-28   0.000 000 0013 e-28   m^2
788: triton-electron mass ratio                             5496.921 5267         0.000 0050
789: triton g factor                                        5.957 924 896         0.000 000 076
790: triton mag. mom.                                       1.504 609 447 e-26    0.000 000 038 e-26    J T^-1
791: triton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3
792: triton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038
793: triton mass                                            5.007 356 30 e-27     0.000 000 22 e-27     kg
794: triton mass energy equivalent                          4.500 387 41 e-10     0.000 000 20 e-10     J
795: triton mass energy equivalent in MeV                   2808.921 005          0.000 062             MeV
796: triton mass in u                                       3.015 500 7134        0.000 000 0025        u
797: triton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1
798: triton-proton mass ratio                               2.993 717 0308        0.000 000 0025
799: unified atomic mass unit                               1.660 538 921 e-27    0.000 000 073 e-27    kg
800: von Klitzing constant                                  25 812.807 4434       0.000 0084            ohm
801: weak mixing angle                                      0.2223                0.0021
802: Wien frequency displacement law constant               5.878 9254 e10        0.000 0053 e10        Hz K^-1
803: Wien wavelength displacement law constant              2.897 7721 e-3        0.000 0026 e-3        m K'''
804: 
805: txt2014 = '''\
806: {220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m
807: alpha particle-electron mass ratio                     7294.299 541 36       0.000 000 24
808: alpha particle mass                                    6.644 657 230 e-27    0.000 000 082 e-27    kg
809: alpha particle mass energy equivalent                  5.971 920 097 e-10    0.000 000 073 e-10    J
810: alpha particle mass energy equivalent in MeV           3727.379 378          0.000 023             MeV
811: alpha particle mass in u                               4.001 506 179 127     0.000 000 000 063     u
812: alpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 063 e-3 kg mol^-1
813: alpha particle-proton mass ratio                       3.972 599 689 07      0.000 000 000 36
814: Angstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m
815: atomic mass constant                                   1.660 539 040 e-27    0.000 000 020 e-27    kg
816: atomic mass constant energy equivalent                 1.492 418 062 e-10    0.000 000 018 e-10    J
817: atomic mass constant energy equivalent in MeV          931.494 0954          0.000 0057            MeV
818: atomic mass unit-electron volt relationship            931.494 0954 e6       0.000 0057 e6         eV
819: atomic mass unit-hartree relationship                  3.423 177 6902 e7     0.000 000 0016 e7     E_h
820: atomic mass unit-hertz relationship                    2.252 342 7206 e23    0.000 000 0010 e23    Hz
821: atomic mass unit-inverse meter relationship            7.513 006 6166 e14    0.000 000 0034 e14    m^-1
822: atomic mass unit-joule relationship                    1.492 418 062 e-10    0.000 000 018 e-10    J
823: atomic mass unit-kelvin relationship                   1.080 954 38 e13      0.000 000 62 e13      K
824: atomic mass unit-kilogram relationship                 1.660 539 040 e-27    0.000 000 020 e-27    kg
825: atomic unit of 1st hyperpolarizability                 3.206 361 329 e-53    0.000 000 020 e-53    C^3 m^3 J^-2
826: atomic unit of 2nd hyperpolarizability                 6.235 380 085 e-65    0.000 000 077 e-65    C^4 m^4 J^-3
827: atomic unit of action                                  1.054 571 800 e-34    0.000 000 013 e-34    J s
828: atomic unit of charge                                  1.602 176 6208 e-19   0.000 000 0098 e-19   C
829: atomic unit of charge density                          1.081 202 3770 e12    0.000 000 0067 e12    C m^-3
830: atomic unit of current                                 6.623 618 183 e-3     0.000 000 041 e-3     A
831: atomic unit of electric dipole mom.                    8.478 353 552 e-30    0.000 000 052 e-30    C m
832: atomic unit of electric field                          5.142 206 707 e11     0.000 000 032 e11     V m^-1
833: atomic unit of electric field gradient                 9.717 362 356 e21     0.000 000 060 e21     V m^-2
834: atomic unit of electric polarizability                 1.648 777 2731 e-41   0.000 000 0011 e-41   C^2 m^2 J^-1
835: atomic unit of electric potential                      27.211 386 02         0.000 000 17          V
836: atomic unit of electric quadrupole mom.                4.486 551 484 e-40    0.000 000 028 e-40    C m^2
837: atomic unit of energy                                  4.359 744 650 e-18    0.000 000 054 e-18    J
838: atomic unit of force                                   8.238 723 36 e-8      0.000 000 10 e-8      N
839: atomic unit of length                                  0.529 177 210 67 e-10 0.000 000 000 12 e-10 m
840: atomic unit of mag. dipole mom.                        1.854 801 999 e-23    0.000 000 011 e-23    J T^-1
841: atomic unit of mag. flux density                       2.350 517 550 e5      0.000 000 014 e5      T
842: atomic unit of magnetizability                         7.891 036 5886 e-29   0.000 000 0090 e-29   J T^-2
843: atomic unit of mass                                    9.109 383 56 e-31     0.000 000 11 e-31     kg
844: atomic unit of mom.um                                  1.992 851 882 e-24    0.000 000 024 e-24    kg m s^-1
845: atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
846: atomic unit of time                                    2.418 884 326509e-17  0.000 000 000014e-17  s
847: atomic unit of velocity                                2.187 691 262 77 e6   0.000 000 000 50 e6   m s^-1
848: Avogadro constant                                      6.022 140 857 e23     0.000 000 074 e23     mol^-1
849: Bohr magneton                                          927.400 9994 e-26     0.000 0057 e-26       J T^-1
850: Bohr magneton in eV/T                                  5.788 381 8012 e-5    0.000 000 0026 e-5    eV T^-1
851: Bohr magneton in Hz/T                                  13.996 245 042 e9     0.000 000 086 e9      Hz T^-1
852: Bohr magneton in inverse meters per tesla              46.686 448 14         0.000 000 29          m^-1 T^-1
853: Bohr magneton in K/T                                   0.671 714 05          0.000 000 39          K T^-1
854: Bohr radius                                            0.529 177 210 67 e-10 0.000 000 000 12 e-10 m
855: Boltzmann constant                                     1.380 648 52 e-23     0.000 000 79 e-23     J K^-1
856: Boltzmann constant in eV/K                             8.617 3303 e-5        0.000 0050 e-5        eV K^-1
857: Boltzmann constant in Hz/K                             2.083 6612 e10        0.000 0012 e10        Hz K^-1
858: Boltzmann constant in inverse meters per kelvin        69.503 457            0.000 040             m^-1 K^-1
859: characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
860: classical electron radius                              2.817 940 3227 e-15   0.000 000 0019 e-15   m
861: Compton wavelength                                     2.426 310 2367 e-12   0.000 000 0011 e-12   m
862: Compton wavelength over 2 pi                           386.159 267 64 e-15   0.000 000 18 e-15     m
863: conductance quantum                                    7.748 091 7310 e-5    0.000 000 0018 e-5    S
864: conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
865: conventional value of von Klitzing constant            25 812.807            (exact)               ohm
866: Cu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m
867: deuteron-electron mag. mom. ratio                      -4.664 345 535 e-4    0.000 000 026 e-4
868: deuteron-electron mass ratio                           3670.482 967 85       0.000 000 13
869: deuteron g factor                                      0.857 438 2311        0.000 000 0048
870: deuteron mag. mom.                                     0.433 073 5040 e-26   0.000 000 0036 e-26   J T^-1
871: deuteron mag. mom. to Bohr magneton ratio              0.466 975 4554 e-3    0.000 000 0026 e-3
872: deuteron mag. mom. to nuclear magneton ratio           0.857 438 2311        0.000 000 0048
873: deuteron mass                                          3.343 583 719 e-27    0.000 000 041 e-27    kg
874: deuteron mass energy equivalent                        3.005 063 183 e-10    0.000 000 037 e-10    J
875: deuteron mass energy equivalent in MeV                 1875.612 928          0.000 012             MeV
876: deuteron mass in u                                     2.013 553 212 745     0.000 000 000 040     u
877: deuteron molar mass                                    2.013 553 212 745 e-3 0.000 000 000 040 e-3 kg mol^-1
878: deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
879: deuteron-proton mag. mom. ratio                        0.307 012 2077        0.000 000 0015
880: deuteron-proton mass ratio                             1.999 007 500 87      0.000 000 000 19
881: deuteron rms charge radius                             2.1413 e-15           0.0025 e-15           m
882: electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
883: electron charge to mass quotient                       -1.758 820 024 e11    0.000 000 011 e11     C kg^-1
884: electron-deuteron mag. mom. ratio                      -2143.923 499         0.000 012
885: electron-deuteron mass ratio                           2.724 437 107 484 e-4 0.000 000 000 096 e-4
886: electron g factor                                      -2.002 319 304 361 82 0.000 000 000 000 52
887: electron gyromag. ratio                                1.760 859 644 e11     0.000 000 011 e11     s^-1 T^-1
888: electron gyromag. ratio over 2 pi                      28 024.951 64         0.000 17              MHz T^-1
889: electron-helion mass ratio                             1.819 543 074 854 e-4 0.000 000 000 088 e-4
890: electron mag. mom.                                     -928.476 4620 e-26    0.000 0057 e-26       J T^-1
891: electron mag. mom. anomaly                             1.159 652 180 91 e-3  0.000 000 000 26 e-3
892: electron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 91 0.000 000 000 000 26
893: electron mag. mom. to nuclear magneton ratio           -1838.281 972 34      0.000 000 17
894: electron mass                                          9.109 383 56 e-31     0.000 000 11 e-31     kg
895: electron mass energy equivalent                        8.187 105 65 e-14     0.000 000 10 e-14     J
896: electron mass energy equivalent in MeV                 0.510 998 9461        0.000 000 0031        MeV
897: electron mass in u                                     5.485 799 090 70 e-4  0.000 000 000 16 e-4  u
898: electron molar mass                                    5.485 799 090 70 e-7  0.000 000 000 16 e-7  kg mol^-1
899: electron-muon mag. mom. ratio                          206.766 9880          0.000 0046
900: electron-muon mass ratio                               4.836 331 70 e-3      0.000 000 11 e-3
901: electron-neutron mag. mom. ratio                       960.920 50            0.000 23
902: electron-neutron mass ratio                            5.438 673 4428 e-4    0.000 000 0027 e-4
903: electron-proton mag. mom. ratio                        -658.210 6866         0.000 0020
904: electron-proton mass ratio                             5.446 170 213 52 e-4  0.000 000 000 52 e-4
905: electron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4
906: electron to alpha particle mass ratio                  1.370 933 554 798 e-4 0.000 000 000 045 e-4
907: electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
908: electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
909: electron-triton mass ratio                             1.819 200 062 203 e-4 0.000 000 000 084 e-4
910: electron volt                                          1.602 176 6208 e-19   0.000 000 0098 e-19   J
911: electron volt-atomic mass unit relationship            1.073 544 1105 e-9    0.000 000 0066 e-9    u
912: electron volt-hartree relationship                     3.674 932 248 e-2     0.000 000 023 e-2     E_h
913: electron volt-hertz relationship                       2.417 989 262 e14     0.000 000 015 e14     Hz
914: electron volt-inverse meter relationship               8.065 544 005 e5      0.000 000 050 e5      m^-1
915: electron volt-joule relationship                       1.602 176 6208 e-19   0.000 000 0098 e-19   J
916: electron volt-kelvin relationship                      1.160 452 21 e4       0.000 000 67 e4       K
917: electron volt-kilogram relationship                    1.782 661 907 e-36    0.000 000 011 e-36    kg
918: elementary charge                                      1.602 176 6208 e-19   0.000 000 0098 e-19   C
919: elementary charge over h                               2.417 989 262 e14     0.000 000 015 e14     A J^-1
920: Faraday constant                                       96 485.332 89         0.000 59              C mol^-1
921: Faraday constant for conventional electric current     96 485.3251           0.0012                C_90 mol^-1
922: Fermi coupling constant                                1.166 3787 e-5        0.000 0006 e-5        GeV^-2
923: fine-structure constant                                7.297 352 5664 e-3    0.000 000 0017 e-3
924: first radiation constant                               3.741 771 790 e-16    0.000 000 046 e-16    W m^2
925: first radiation constant for spectral radiance         1.191 042 953 e-16    0.000 000 015 e-16    W m^2 sr^-1
926: hartree-atomic mass unit relationship                  2.921 262 3197 e-8    0.000 000 0013 e-8    u
927: hartree-electron volt relationship                     27.211 386 02         0.000 000 17          eV
928: Hartree energy                                         4.359 744 650 e-18    0.000 000 054 e-18    J
929: Hartree energy in eV                                   27.211 386 02         0.000 000 17          eV
930: hartree-hertz relationship                             6.579 683 920 711 e15 0.000 000 000 039 e15 Hz
931: hartree-inverse meter relationship                     2.194 746 313 702 e7  0.000 000 000 013 e7  m^-1
932: hartree-joule relationship                             4.359 744 650 e-18    0.000 000 054 e-18    J
933: hartree-kelvin relationship                            3.157 7513 e5         0.000 0018 e5         K
934: hartree-kilogram relationship                          4.850 870 129 e-35    0.000 000 060 e-35    kg
935: helion-electron mass ratio                             5495.885 279 22       0.000 000 27
936: helion g factor                                        -4.255 250 616        0.000 000 050
937: helion mag. mom.                                       -1.074 617 522 e-26   0.000 000 014 e-26    J T^-1
938: helion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3
939: helion mag. mom. to nuclear magneton ratio             -2.127 625 308        0.000 000 025
940: helion mass                                            5.006 412 700 e-27    0.000 000 062 e-27    kg
941: helion mass energy equivalent                          4.499 539 341 e-10    0.000 000 055 e-10    J
942: helion mass energy equivalent in MeV                   2808.391 586          0.000 017             MeV
943: helion mass in u                                       3.014 932 246 73      0.000 000 000 12      u
944: helion molar mass                                      3.014 932 246 73 e-3  0.000 000 000 12 e-3  kg mol^-1
945: helion-proton mass ratio                               2.993 152 670 46      0.000 000 000 29
946: hertz-atomic mass unit relationship                    4.439 821 6616 e-24   0.000 000 0020 e-24   u
947: hertz-electron volt relationship                       4.135 667 662 e-15    0.000 000 025 e-15    eV
948: hertz-hartree relationship                             1.5198298460088 e-16  0.0000000000090e-16   E_h
949: hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
950: hertz-joule relationship                               6.626 070 040 e-34    0.000 000 081 e-34    J
951: hertz-kelvin relationship                              4.799 2447 e-11       0.000 0028 e-11       K
952: hertz-kilogram relationship                            7.372 497 201 e-51    0.000 000 091 e-51    kg
953: inverse fine-structure constant                        137.035 999 139       0.000 000 031
954: inverse meter-atomic mass unit relationship            1.331 025 049 00 e-15 0.000 000 000 61 e-15 u
955: inverse meter-electron volt relationship               1.239 841 9739 e-6    0.000 000 0076 e-6    eV
956: inverse meter-hartree relationship                     4.556 335 252 767 e-8 0.000 000 000 027 e-8 E_h
957: inverse meter-hertz relationship                       299 792 458           (exact)               Hz
958: inverse meter-joule relationship                       1.986 445 824 e-25    0.000 000 024 e-25    J
959: inverse meter-kelvin relationship                      1.438 777 36 e-2      0.000 000 83 e-2      K
960: inverse meter-kilogram relationship                    2.210 219 057 e-42    0.000 000 027 e-42    kg
961: inverse of conductance quantum                         12 906.403 7278       0.000 0029            ohm
962: Josephson constant                                     483 597.8525 e9       0.0030 e9             Hz V^-1
963: joule-atomic mass unit relationship                    6.700 535 363 e9      0.000 000 082 e9      u
964: joule-electron volt relationship                       6.241 509 126 e18     0.000 000 038 e18     eV
965: joule-hartree relationship                             2.293 712 317 e17     0.000 000 028 e17     E_h
966: joule-hertz relationship                               1.509 190 205 e33     0.000 000 019 e33     Hz
967: joule-inverse meter relationship                       5.034 116 651 e24     0.000 000 062 e24     m^-1
968: joule-kelvin relationship                              7.242 9731 e22        0.000 0042 e22        K
969: joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
970: kelvin-atomic mass unit relationship                   9.251 0842 e-14       0.000 0053 e-14       u
971: kelvin-electron volt relationship                      8.617 3303 e-5        0.000 0050 e-5        eV
972: kelvin-hartree relationship                            3.166 8105 e-6        0.000 0018 e-6        E_h
973: kelvin-hertz relationship                              2.083 6612 e10        0.000 0012 e10        Hz
974: kelvin-inverse meter relationship                      69.503 457            0.000 040             m^-1
975: kelvin-joule relationship                              1.380 648 52 e-23     0.000 000 79 e-23     J
976: kelvin-kilogram relationship                           1.536 178 65 e-40     0.000 000 88 e-40     kg
977: kilogram-atomic mass unit relationship                 6.022 140 857 e26     0.000 000 074 e26     u
978: kilogram-electron volt relationship                    5.609 588 650 e35     0.000 000 034 e35     eV
979: kilogram-hartree relationship                          2.061 485 823 e34     0.000 000 025 e34     E_h
980: kilogram-hertz relationship                            1.356 392 512 e50     0.000 000 017 e50     Hz
981: kilogram-inverse meter relationship                    4.524 438 411 e41     0.000 000 056 e41     m^-1
982: kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
983: kilogram-kelvin relationship                           6.509 6595 e39        0.000 0037 e39        K
984: lattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m
985: Loschmidt constant (273.15 K, 100 kPa)                 2.651 6467 e25        0.000 0015 e25        m^-3
986: Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7811 e25        0.000 0015 e25        m^-3
987: mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
988: mag. flux quantum                                      2.067 833 831 e-15    0.000 000 013 e-15    Wb
989: molar gas constant                                     8.314 4598            0.000 0048            J mol^-1 K^-1
990: molar mass constant                                    1 e-3                 (exact)               kg mol^-1
991: molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
992: molar Planck constant                                  3.990 312 7110 e-10   0.000 000 0018 e-10   J s mol^-1
993: molar Planck constant times c                          0.119 626 565 582     0.000 000 000 054     J m mol^-1
994: molar volume of ideal gas (273.15 K, 100 kPa)          22.710 947 e-3        0.000 013 e-3         m^3 mol^-1
995: molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 962 e-3        0.000 013 e-3         m^3 mol^-1
996: molar volume of silicon                                12.058 832 14 e-6     0.000 000 61 e-6      m^3 mol^-1
997: Mo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m
998: muon Compton wavelength                                11.734 441 11 e-15    0.000 000 26 e-15     m
999: muon Compton wavelength over 2 pi                      1.867 594 308 e-15    0.000 000 042 e-15    m
1000: muon-electron mass ratio                               206.768 2826          0.000 0046
1001: muon g factor                                          -2.002 331 8418       0.000 000 0013
1002: muon mag. mom.                                         -4.490 448 26 e-26    0.000 000 10 e-26     J T^-1
1003: muon mag. mom. anomaly                                 1.165 920 89 e-3      0.000 000 63 e-3
1004: muon mag. mom. to Bohr magneton ratio                  -4.841 970 48 e-3     0.000 000 11 e-3
1005: muon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 20
1006: muon mass                                              1.883 531 594 e-28    0.000 000 048 e-28    kg
1007: muon mass energy equivalent                            1.692 833 774 e-11    0.000 000 043 e-11    J
1008: muon mass energy equivalent in MeV                     105.658 3745          0.000 0024            MeV
1009: muon mass in u                                         0.113 428 9257        0.000 000 0025        u
1010: muon molar mass                                        0.113 428 9257 e-3    0.000 000 0025 e-3    kg mol^-1
1011: muon-neutron mass ratio                                0.112 454 5167        0.000 000 0025
1012: muon-proton mag. mom. ratio                            -3.183 345 142        0.000 000 071
1013: muon-proton mass ratio                                 0.112 609 5262        0.000 000 0025
1014: muon-tau mass ratio                                    5.946 49 e-2          0.000 54 e-2
1015: natural unit of action                                 1.054 571 800 e-34    0.000 000 013 e-34    J s
1016: natural unit of action in eV s                         6.582 119 514 e-16    0.000 000 040 e-16    eV s
1017: natural unit of energy                                 8.187 105 65 e-14     0.000 000 10 e-14     J
1018: natural unit of energy in MeV                          0.510 998 9461        0.000 000 0031        MeV
1019: natural unit of length                                 386.159 267 64 e-15   0.000 000 18 e-15     m
1020: natural unit of mass                                   9.109 383 56 e-31     0.000 000 11 e-31     kg
1021: natural unit of mom.um                                 2.730 924 488 e-22    0.000 000 034 e-22    kg m s^-1
1022: natural unit of mom.um in MeV/c                        0.510 998 9461        0.000 000 0031        MeV/c
1023: natural unit of time                                   1.288 088 667 12 e-21 0.000 000 000 58 e-21 s
1024: natural unit of velocity                               299 792 458           (exact)               m s^-1
1025: neutron Compton wavelength                             1.319 590 904 81 e-15 0.000 000 000 88 e-15 m
1026: neutron Compton wavelength over 2 pi                   0.210 019 415 36 e-15 0.000 000 000 14 e-15 m
1027: neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
1028: neutron-electron mass ratio                            1838.683 661 58       0.000 000 90
1029: neutron g factor                                       -3.826 085 45         0.000 000 90
1030: neutron gyromag. ratio                                 1.832 471 72 e8       0.000 000 43 e8       s^-1 T^-1
1031: neutron gyromag. ratio over 2 pi                       29.164 6933           0.000 0069            MHz T^-1
1032: neutron mag. mom.                                      -0.966 236 50 e-26    0.000 000 23 e-26     J T^-1
1033: neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
1034: neutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45
1035: neutron mass                                           1.674 927 471 e-27    0.000 000 021 e-27    kg
1036: neutron mass energy equivalent                         1.505 349 739 e-10    0.000 000 019 e-10    J
1037: neutron mass energy equivalent in MeV                  939.565 4133          0.000 0058            MeV
1038: neutron mass in u                                      1.008 664 915 88      0.000 000 000 49      u
1039: neutron molar mass                                     1.008 664 915 88 e-3  0.000 000 000 49 e-3  kg mol^-1
1040: neutron-muon mass ratio                                8.892 484 08          0.000 000 20
1041: neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
1042: neutron-proton mass difference                         2.305 573 77 e-30     0.000 000 85 e-30
1043: neutron-proton mass difference energy equivalent       2.072 146 37 e-13     0.000 000 76 e-13
1044: neutron-proton mass difference energy equivalent in MeV 1.293 332 05         0.000 000 48
1045: neutron-proton mass difference in u                    0.001 388 449 00      0.000 000 000 51
1046: neutron-proton mass ratio                              1.001 378 418 98      0.000 000 000 51
1047: neutron-tau mass ratio                                 0.528 790             0.000 048
1048: neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
1049: Newtonian constant of gravitation                      6.674 08 e-11         0.000 31 e-11         m^3 kg^-1 s^-2
1050: Newtonian constant of gravitation over h-bar c         6.708 61 e-39         0.000 31 e-39         (GeV/c^2)^-2
1051: nuclear magneton                                       5.050 783 699 e-27    0.000 000 031 e-27    J T^-1
1052: nuclear magneton in eV/T                               3.152 451 2550 e-8    0.000 000 0015 e-8    eV T^-1
1053: nuclear magneton in inverse meters per tesla           2.542 623 432 e-2     0.000 000 016 e-2     m^-1 T^-1
1054: nuclear magneton in K/T                                3.658 2690 e-4        0.000 0021 e-4        K T^-1
1055: nuclear magneton in MHz/T                              7.622 593 285         0.000 000 047         MHz T^-1
1056: Planck constant                                        6.626 070 040 e-34    0.000 000 081 e-34    J s
1057: Planck constant in eV s                                4.135 667 662 e-15    0.000 000 025 e-15    eV s
1058: Planck constant over 2 pi                              1.054 571 800 e-34    0.000 000 013 e-34    J s
1059: Planck constant over 2 pi in eV s                      6.582 119 514 e-16    0.000 000 040 e-16    eV s
1060: Planck constant over 2 pi times c in MeV fm            197.326 9788          0.000 0012            MeV fm
1061: Planck length                                          1.616 229 e-35        0.000 038 e-35        m
1062: Planck mass                                            2.176 470 e-8         0.000 051 e-8         kg
1063: Planck mass energy equivalent in GeV                   1.220 910 e19         0.000 029 e19         GeV
1064: Planck temperature                                     1.416 808 e32         0.000 033 e32         K
1065: Planck time                                            5.391 16 e-44         0.000 13 e-44         s
1066: proton charge to mass quotient                         9.578 833 226 e7      0.000 000 059 e7      C kg^-1
1067: proton Compton wavelength                              1.321 409 853 96 e-15 0.000 000 000 61 e-15 m
1068: proton Compton wavelength over 2 pi                    0.210 308910109e-15   0.000 000 000097e-15  m
1069: proton-electron mass ratio                             1836.152 673 89       0.000 000 17
1070: proton g factor                                        5.585 694 702         0.000 000 017
1071: proton gyromag. ratio                                  2.675 221 900 e8      0.000 000 018 e8      s^-1 T^-1
1072: proton gyromag. ratio over 2 pi                        42.577 478 92         0.000 000 29          MHz T^-1
1073: proton mag. mom.                                       1.410 606 7873 e-26   0.000 000 0097 e-26   J T^-1
1074: proton mag. mom. to Bohr magneton ratio                1.521 032 2053 e-3    0.000 000 0046 e-3
1075: proton mag. mom. to nuclear magneton ratio             2.792 847 3508        0.000 000 0085
1076: proton mag. shielding correction                       25.691 e-6            0.011 e-6
1077: proton mass                                            1.672 621 898 e-27    0.000 000 021 e-27    kg
1078: proton mass energy equivalent                          1.503 277 593 e-10    0.000 000 018 e-10    J
1079: proton mass energy equivalent in MeV                   938.272 0813          0.000 0058            MeV
1080: proton mass in u                                       1.007 276 466 879     0.000 000 000 091     u
1081: proton molar mass                                      1.007 276 466 879 e-3 0.000 000 000 091 e-3 kg mol^-1
1082: proton-muon mass ratio                                 8.880 243 38          0.000 000 20
1083: proton-neutron mag. mom. ratio                         -1.459 898 05         0.000 000 34
1084: proton-neutron mass ratio                              0.998 623 478 44      0.000 000 000 51
1085: proton rms charge radius                               0.8751 e-15           0.0061 e-15           m
1086: proton-tau mass ratio                                  0.528 063             0.000 048
1087: quantum of circulation                                 3.636 947 5486 e-4    0.000 000 0017 e-4    m^2 s^-1
1088: quantum of circulation times 2                         7.273 895 0972 e-4    0.000 000 0033 e-4    m^2 s^-1
1089: Rydberg constant                                       10 973 731.568 508    0.000 065             m^-1
1090: Rydberg constant times c in Hz                         3.289 841 960 355 e15 0.000 000 000 019 e15 Hz
1091: Rydberg constant times hc in eV                        13.605 693 009        0.000 000 084         eV
1092: Rydberg constant times hc in J                         2.179 872 325 e-18    0.000 000 027 e-18    J
1093: Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7084           0.000 0014
1094: Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8714           0.000 0014
1095: second radiation constant                              1.438 777 36 e-2      0.000 000 83 e-2      m K
1096: shielded helion gyromag. ratio                         2.037 894 585 e8      0.000 000 027 e8      s^-1 T^-1
1097: shielded helion gyromag. ratio over 2 pi               32.434 099 66         0.000 000 43          MHz T^-1
1098: shielded helion mag. mom.                              -1.074 553 080 e-26   0.000 000 014 e-26    J T^-1
1099: shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
1100: shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 720        0.000 000 025
1101: shielded helion to proton mag. mom. ratio              -0.761 766 5603       0.000 000 0092
1102: shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
1103: shielded proton gyromag. ratio                         2.675 153 171 e8      0.000 000 033 e8      s^-1 T^-1
1104: shielded proton gyromag. ratio over 2 pi               42.576 385 07         0.000 000 53          MHz T^-1
1105: shielded proton mag. mom.                              1.410 570 547 e-26    0.000 000 018 e-26    J T^-1
1106: shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
1107: shielded proton mag. mom. to nuclear magneton ratio    2.792 775 600         0.000 000 030
1108: speed of light in vacuum                               299 792 458           (exact)               m s^-1
1109: standard acceleration of gravity                       9.806 65              (exact)               m s^-2
1110: standard atmosphere                                    101 325               (exact)               Pa
1111: standard-state pressure                                100 000               (exact)               Pa
1112: Stefan-Boltzmann constant                              5.670 367 e-8         0.000 013 e-8         W m^-2 K^-4
1113: tau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m
1114: tau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m
1115: tau-electron mass ratio                                3477.15               0.31
1116: tau mass                                               3.167 47 e-27         0.000 29 e-27         kg
1117: tau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J
1118: tau mass energy equivalent in MeV                      1776.82               0.16                  MeV
1119: tau mass in u                                          1.907 49              0.000 17              u
1120: tau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1
1121: tau-muon mass ratio                                    16.8167               0.0015
1122: tau-neutron mass ratio                                 1.891 11              0.000 17
1123: tau-proton mass ratio                                  1.893 72              0.000 17
1124: Thomson cross section                                  0.665 245 871 58 e-28 0.000 000 000 91 e-28 m^2
1125: triton-electron mass ratio                             5496.921 535 88       0.000 000 26
1126: triton g factor                                        5.957 924 920         0.000 000 028
1127: triton mag. mom.                                       1.504 609 503 e-26    0.000 000 012 e-26    J T^-1
1128: triton mag. mom. to Bohr magneton ratio                1.622 393 6616 e-3    0.000 000 0076 e-3
1129: triton mag. mom. to nuclear magneton ratio             2.978 962 460         0.000 000 014
1130: triton mass                                            5.007 356 665 e-27    0.000 000 062 e-27    kg
1131: triton mass energy equivalent                          4.500 387 735 e-10    0.000 000 055 e-10    J
1132: triton mass energy equivalent in MeV                   2808.921 112          0.000 017             MeV
1133: triton mass in u                                       3.015 500 716 32      0.000 000 000 11      u
1134: triton molar mass                                      3.015 500 716 32 e-3  0.000 000 000 11 e-3  kg mol^-1
1135: triton-proton mass ratio                               2.993 717 033 48      0.000 000 000 22
1136: unified atomic mass unit                               1.660 539 040 e-27    0.000 000 020 e-27    kg
1137: von Klitzing constant                                  25 812.807 4555       0.000 0059            ohm
1138: weak mixing angle                                      0.2223                0.0021
1139: Wien frequency displacement law constant               5.878 9238 e10        0.000 0034 e10        Hz K^-1
1140: Wien wavelength displacement law constant              2.897 7729 e-3        0.000 0017 e-3        m K'''
1141: 
1142: # -----------------------------------------------------------------------------
1143: 
1144: physical_constants = {}
1145: 
1146: 
1147: def parse_constants(d):
1148:     constants = {}
1149:     for line in d.split('\n'):
1150:         name = line[:55].rstrip()
1151:         val = line[55:77].replace(' ', '').replace('...', '')
1152:         val = float(val)
1153:         uncert = line[77:99].replace(' ', '').replace('(exact)', '0')
1154:         uncert = float(uncert)
1155:         units = line[99:].rstrip()
1156:         constants[name] = (val, units, uncert)
1157:     return constants
1158: 
1159: 
1160: _physical_constants_2002 = parse_constants(txt2002)
1161: _physical_constants_2006 = parse_constants(txt2006)
1162: _physical_constants_2010 = parse_constants(txt2010)
1163: _physical_constants_2014 = parse_constants(txt2014)
1164: 
1165: 
1166: physical_constants.update(_physical_constants_2002)
1167: physical_constants.update(_physical_constants_2006)
1168: physical_constants.update(_physical_constants_2010)
1169: physical_constants.update(_physical_constants_2014)
1170: _current_constants = _physical_constants_2014
1171: _current_codata = "CODATA 2014"
1172: 
1173: # check obsolete values
1174: _obsolete_constants = {}
1175: for k in physical_constants:
1176:     if k not in _current_constants:
1177:         _obsolete_constants[k] = True
1178: 
1179: # generate some additional aliases
1180: _aliases = {}
1181: for k in _physical_constants_2002:
1182:     if 'magn.' in k:
1183:         _aliases[k] = k.replace('magn.', 'mag.')
1184: for k in _physical_constants_2006:
1185:     if 'momentum' in k:
1186:         _aliases[k] = k.replace('momentum', 'mom.um')
1187: 
1188: 
1189: class ConstantWarning(DeprecationWarning):
1190:     '''Accessing a constant no longer in current CODATA data set'''
1191:     pass
1192: 
1193: 
1194: def _check_obsolete(key):
1195:     if key in _obsolete_constants and key not in _aliases:
1196:         warnings.warn("Constant '%s' is not in current %s data set" % (
1197:             key, _current_codata), ConstantWarning)
1198: 
1199: 
1200: def value(key):
1201:     '''
1202:     Value in physical_constants indexed by key
1203: 
1204:     Parameters
1205:     ----------
1206:     key : Python string or unicode
1207:         Key in dictionary `physical_constants`
1208: 
1209:     Returns
1210:     -------
1211:     value : float
1212:         Value in `physical_constants` corresponding to `key`
1213: 
1214:     See Also
1215:     --------
1216:     codata : Contains the description of `physical_constants`, which, as a
1217:         dictionary literal object, does not itself possess a docstring.
1218: 
1219:     Examples
1220:     --------
1221:     >>> from scipy import constants
1222:     >>> constants.value(u'elementary charge')
1223:         1.6021766208e-19
1224: 
1225:     '''
1226:     _check_obsolete(key)
1227:     return physical_constants[key][0]
1228: 
1229: 
1230: def unit(key):
1231:     '''
1232:     Unit in physical_constants indexed by key
1233: 
1234:     Parameters
1235:     ----------
1236:     key : Python string or unicode
1237:         Key in dictionary `physical_constants`
1238: 
1239:     Returns
1240:     -------
1241:     unit : Python string
1242:         Unit in `physical_constants` corresponding to `key`
1243: 
1244:     See Also
1245:     --------
1246:     codata : Contains the description of `physical_constants`, which, as a
1247:         dictionary literal object, does not itself possess a docstring.
1248: 
1249:     Examples
1250:     --------
1251:     >>> from scipy import constants
1252:     >>> constants.unit(u'proton mass')
1253:     'kg'
1254: 
1255:     '''
1256:     _check_obsolete(key)
1257:     return physical_constants[key][1]
1258: 
1259: 
1260: def precision(key):
1261:     '''
1262:     Relative precision in physical_constants indexed by key
1263: 
1264:     Parameters
1265:     ----------
1266:     key : Python string or unicode
1267:         Key in dictionary `physical_constants`
1268: 
1269:     Returns
1270:     -------
1271:     prec : float
1272:         Relative precision in `physical_constants` corresponding to `key`
1273: 
1274:     See Also
1275:     --------
1276:     codata : Contains the description of `physical_constants`, which, as a
1277:         dictionary literal object, does not itself possess a docstring.
1278: 
1279:     Examples
1280:     --------
1281:     >>> from scipy import constants
1282:     >>> constants.precision(u'proton mass')
1283:     1.2555138746605121e-08
1284: 
1285:     '''
1286:     _check_obsolete(key)
1287:     return physical_constants[key][2] / physical_constants[key][0]
1288: 
1289: 
1290: def find(sub=None, disp=False):
1291:     '''
1292:     Return list of physical_constant keys containing a given string.
1293: 
1294:     Parameters
1295:     ----------
1296:     sub : str, unicode
1297:         Sub-string to search keys for.  By default, return all keys.
1298:     disp : bool
1299:         If True, print the keys that are found, and return None.
1300:         Otherwise, return the list of keys without printing anything.
1301: 
1302:     Returns
1303:     -------
1304:     keys : list or None
1305:         If `disp` is False, the list of keys is returned.
1306:         Otherwise, None is returned.
1307: 
1308:     See Also
1309:     --------
1310:     codata : Contains the description of `physical_constants`, which, as a
1311:         dictionary literal object, does not itself possess a docstring.
1312: 
1313:     Examples
1314:     --------
1315:     >>> from scipy.constants import find, physical_constants
1316: 
1317:     Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?
1318: 
1319:     >>> find('boltzmann')
1320:     ['Boltzmann constant',
1321:      'Boltzmann constant in Hz/K',
1322:      'Boltzmann constant in eV/K',
1323:      'Boltzmann constant in inverse meters per kelvin',
1324:      'Stefan-Boltzmann constant']
1325: 
1326:     Get the constant called 'Boltzmann constant in Hz/K':
1327: 
1328:     >>> physical_constants['Boltzmann constant in Hz/K']
1329:     (20836612000.0, 'Hz K^-1', 12000.0)
1330: 
1331:     Find constants with 'radius' in the key:
1332: 
1333:     >>> find('radius')
1334:     ['Bohr radius',
1335:      'classical electron radius',
1336:      'deuteron rms charge radius',
1337:      'proton rms charge radius']
1338:     >>> physical_constants['classical electron radius']
1339:     (2.8179403227e-15, 'm', 1.9e-24)
1340: 
1341:     '''
1342:     if sub is None:
1343:         result = list(_current_constants.keys())
1344:     else:
1345:         result = [key for key in _current_constants
1346:                   if sub.lower() in key.lower()]
1347: 
1348:     result.sort()
1349:     if disp:
1350:         for key in result:
1351:             print(key)
1352:         return
1353:     else:
1354:         return result
1355: 
1356: # Table is lacking some digits for exact values: calculate from definition
1357: c = value('speed of light in vacuum')
1358: mu0 = 4e-7 * pi
1359: epsilon0 = 1 / (mu0 * c * c)
1360: 
1361: exact_values = {
1362:     'mag. constant': (mu0, 'N A^-2', 0.0),
1363:     'electric constant': (epsilon0, 'F m^-1', 0.0),
1364:     'characteristic impedance of vacuum': (sqrt(mu0 / epsilon0), 'ohm', 0.0),
1365:     'atomic unit of permittivity': (4 * epsilon0 * pi, 'F m^-1', 0.0),
1366:     'joule-kilogram relationship': (1 / (c * c), 'kg', 0.0),
1367:     'kilogram-joule relationship': (c * c, 'J', 0.0),
1368:     'hertz-inverse meter relationship': (1 / c, 'm^-1', 0.0)
1369: }
1370: 
1371: # sanity check
1372: for key in exact_values:
1373:     val = _current_constants[key][0]
1374:     if abs(exact_values[key][0] - val) / val > 1e-9:
1375:         raise ValueError("Constants.codata: exact values too far off.")
1376: 
1377: physical_constants.update(exact_values)
1378: 
1379: # finally, insert aliases for values
1380: for k, v in list(_aliases.items()):
1381:     if v in _current_constants:
1382:         physical_constants[k] = physical_constants[v]
1383:     else:
1384:         del _aliases[k]
1385: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_13465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '\nFundamental Physical Constants\n------------------------------\n\nThese constants are taken from CODATA Recommended Values of the Fundamental\nPhysical Constants 2014.\n\nObject\n------\nphysical_constants : dict\n    A dictionary containing physical constants. Keys are the names of physical\n    constants, values are tuples (value, units, precision).\n\nFunctions\n---------\nvalue(key):\n    Returns the value of the physical constant(key).\nunit(key):\n    Returns the units of the physical constant(key).\nprecision(key):\n    Returns the relative precision of the physical constant(key).\nfind(sub):\n    Prints or returns list of keys containing the string sub, default is all.\n\nSource\n------\nThe values of the constants provided at this site are recommended for\ninternational use by CODATA and are the latest available. Termed the "2014\nCODATA recommended values," they are generally recognized worldwide for use in\nall fields of science and technology. The values became available on 25 June\n2015 and replaced the 2010 CODATA set. They are based on all of the data\navailable through 31 December 2014. The 2014 adjustment was carried out under\nthe auspices of the CODATA Task Group on Fundamental Constants. Also available\nis an introduction to the constants for non-experts at\nhttp://physics.nist.gov/cuu/Constants/introduction.html\n\nReferences\n----------\nTheoretical and experimental publications relevant to the fundamental constants\nand closely related precision measurements published since the mid 1980s, but\nalso including many older papers of particular interest, some of which date\nback to the 1800s. To search bibliography visit\n\nhttp://physics.nist.gov/cuu/Constants/\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'import warnings' statement (line 54)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 0))

# 'from math import pi, sqrt' statement (line 55)
try:
    from math import pi, sqrt

except:
    pi = UndefinedType
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'math', None, module_type_store, ['pi', 'sqrt'], [pi, sqrt])


# Assigning a List to a Name (line 57):
__all__ = ['physical_constants', 'value', 'unit', 'precision', 'find', 'ConstantWarning']
module_type_store.set_exportable_members(['physical_constants', 'value', 'unit', 'precision', 'find', 'ConstantWarning'])

# Obtaining an instance of the builtin type 'list' (line 57)
list_13466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
str_13467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'str', 'physical_constants')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_13466, str_13467)
# Adding element type (line 57)
str_13468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'str', 'value')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_13466, str_13468)
# Adding element type (line 57)
str_13469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'str', 'unit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_13466, str_13469)
# Adding element type (line 57)
str_13470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 50), 'str', 'precision')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_13466, str_13470)
# Adding element type (line 57)
str_13471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 63), 'str', 'find')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_13466, str_13471)
# Adding element type (line 57)
str_13472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 11), 'str', 'ConstantWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_13466, str_13472)

# Assigning a type to the variable '__all__' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '__all__', list_13466)
str_13473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'str', '\nSource:  http://physics.nist.gov/cuu/Constants/index.html\n\nThe values of the constants provided at the above site are recommended for\ninternational use by CODATA and are the latest available. Termed the "2006\nCODATA recommended values", they are generally recognized worldwide for use\nin all fields of science and technology. The values became available in March\n2007 and replaced the 2002 CODATA set. They are based on all of the data\navailable through 31 December 2006. The 2006 adjustment was carried out under\nthe auspices of the CODATA Task Group on Fundamental Constants.\n')

# Assigning a Str to a Name (line 78):
str_13474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', 'Wien displacement law constant                         2.897 7685e-3         0.000 0051e-3         m K\natomic unit of 1st hyperpolarizablity                  3.206 361 51e-53      0.000 000 28e-53      C^3 m^3 J^-2\natomic unit of 2nd hyperpolarizablity                  6.235 3808e-65        0.000 0011e-65        C^4 m^4 J^-3\natomic unit of electric dipole moment                  8.478 353 09e-30      0.000 000 73e-30      C m\natomic unit of electric polarizablity                  1.648 777 274e-41     0.000 000 016e-41     C^2 m^2 J^-1\natomic unit of electric quadrupole moment              4.486 551 24e-40      0.000 000 39e-40      C m^2\natomic unit of magn. dipole moment                     1.854 801 90e-23      0.000 000 16e-23      J T^-1\natomic unit of magn. flux density                      2.350 517 42e5        0.000 000 20e5        T\ndeuteron magn. moment                                  0.433 073 482e-26     0.000 000 038e-26     J T^-1\ndeuteron magn. moment to Bohr magneton ratio           0.466 975 4567e-3     0.000 000 0050e-3\ndeuteron magn. moment to nuclear magneton ratio        0.857 438 2329        0.000 000 0092\ndeuteron-electron magn. moment ratio                   -4.664 345 548e-4     0.000 000 050e-4\ndeuteron-proton magn. moment ratio                     0.307 012 2084        0.000 000 0045\ndeuteron-neutron magn. moment ratio                    -0.448 206 52         0.000 000 11\nelectron gyromagn. ratio                               1.760 859 74e11       0.000 000 15e11       s^-1 T^-1\nelectron gyromagn. ratio over 2 pi                     28 024.9532           0.0024                MHz T^-1\nelectron magn. moment                                  -928.476 412e-26      0.000 080e-26         J T^-1\nelectron magn. moment to Bohr magneton ratio           -1.001 159 652 1859   0.000 000 000 0038\nelectron magn. moment to nuclear magneton ratio        -1838.281 971 07      0.000 000 85\nelectron magn. moment anomaly                          1.159 652 1859e-3     0.000 000 0038e-3\nelectron to shielded proton magn. moment ratio         -658.227 5956         0.000 0071\nelectron to shielded helion magn. moment ratio         864.058 255           0.000 010\nelectron-deuteron magn. moment ratio                   -2143.923 493         0.000 023\nelectron-muon magn. moment ratio                       206.766 9894          0.000 0054\nelectron-neutron magn. moment ratio                    960.920 50            0.000 23\nelectron-proton magn. moment ratio                     -658.210 6862         0.000 0066\nmagn. constant                                         12.566 370 614...e-7  0                     N A^-2\nmagn. flux quantum                                     2.067 833 72e-15      0.000 000 18e-15      Wb\nmuon magn. moment                                      -4.490 447 99e-26     0.000 000 40e-26      J T^-1\nmuon magn. moment to Bohr magneton ratio               -4.841 970 45e-3      0.000 000 13e-3\nmuon magn. moment to nuclear magneton ratio            -8.890 596 98         0.000 000 23\nmuon-proton magn. moment ratio                         -3.183 345 118        0.000 000 089\nneutron gyromagn. ratio                                1.832 471 83e8        0.000 000 46e8        s^-1 T^-1\nneutron gyromagn. ratio over 2 pi                      29.164 6950           0.000 0073            MHz T^-1\nneutron magn. moment                                   -0.966 236 45e-26     0.000 000 24e-26      J T^-1\nneutron magn. moment to Bohr magneton ratio            -1.041 875 63e-3      0.000 000 25e-3\nneutron magn. moment to nuclear magneton ratio         -1.913 042 73         0.000 000 45\nneutron to shielded proton magn. moment ratio          -0.684 996 94         0.000 000 16\nneutron-electron magn. moment ratio                    1.040 668 82e-3       0.000 000 25e-3\nneutron-proton magn. moment ratio                      -0.684 979 34         0.000 000 16\nproton gyromagn. ratio                                 2.675 222 05e8        0.000 000 23e8        s^-1 T^-1\nproton gyromagn. ratio over 2 pi                       42.577 4813           0.000 0037            MHz T^-1\nproton magn. moment                                    1.410 606 71e-26      0.000 000 12e-26      J T^-1\nproton magn. moment to Bohr magneton ratio             1.521 032 206e-3      0.000 000 015e-3\nproton magn. moment to nuclear magneton ratio          2.792 847 351         0.000 000 028\nproton magn. shielding correction                      25.689e-6             0.015e-6\nproton-neutron magn. moment ratio                      -1.459 898 05         0.000 000 34\nshielded helion gyromagn. ratio                        2.037 894 70e8        0.000 000 18e8        s^-1 T^-1\nshielded helion gyromagn. ratio over 2 pi              32.434 1015           0.000 0028            MHz T^-1\nshielded helion magn. moment                           -1.074 553 024e-26    0.000 000 093e-26     J T^-1\nshielded helion magn. moment to Bohr magneton ratio    -1.158 671 474e-3     0.000 000 014e-3\nshielded helion magn. moment to nuclear magneton ratio -2.127 497 723        0.000 000 025\nshielded helion to proton magn. moment ratio           -0.761 766 562        0.000 000 012\nshielded helion to shielded proton magn. moment ratio  -0.761 786 1313       0.000 000 0033\nshielded helion gyromagn. ratio                        2.037 894 70e8        0.000 000 18e8        s^-1 T^-1\nshielded helion gyromagn. ratio over 2 pi              32.434 1015           0.000 0028            MHz T^-1\nshielded proton magn. moment                           1.410 570 47e-26      0.000 000 12e-26      J T^-1\nshielded proton magn. moment to Bohr magneton ratio    1.520 993 132e-3      0.000 000 016e-3\nshielded proton magn. moment to nuclear magneton ratio 2.792 775 604         0.000 000 030\n{220} lattice spacing of silicon                       192.015 5965e-12      0.000 0070e-12        m')
# Assigning a type to the variable 'txt2002' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'txt2002', str_13474)

# Assigning a Str to a Name (line 140):
str_13475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'str', 'lattice spacing of silicon                             192.015 5762 e-12     0.000 0050 e-12       m\nalpha particle-electron mass ratio                     7294.299 5365         0.000 0031\nalpha particle mass                                    6.644 656 20 e-27     0.000 000 33 e-27     kg\nalpha particle mass energy equivalent                  5.971 919 17 e-10     0.000 000 30 e-10     J\nalpha particle mass energy equivalent in MeV           3727.379 109          0.000 093             MeV\nalpha particle mass in u                               4.001 506 179 127     0.000 000 000 062     u\nalpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 062 e-3 kg mol^-1\nalpha particle-proton mass ratio                       3.972 599 689 51      0.000 000 000 41\nAngstrom star                                          1.000 014 98 e-10     0.000 000 90 e-10     m\natomic mass constant                                   1.660 538 782 e-27    0.000 000 083 e-27    kg\natomic mass constant energy equivalent                 1.492 417 830 e-10    0.000 000 074 e-10    J\natomic mass constant energy equivalent in MeV          931.494 028           0.000 023             MeV\natomic mass unit-electron volt relationship            931.494 028 e6        0.000 023 e6          eV\natomic mass unit-hartree relationship                  3.423 177 7149 e7     0.000 000 0049 e7     E_h\natomic mass unit-hertz relationship                    2.252 342 7369 e23    0.000 000 0032 e23    Hz\natomic mass unit-inverse meter relationship            7.513 006 671 e14     0.000 000 011 e14     m^-1\natomic mass unit-joule relationship                    1.492 417 830 e-10    0.000 000 074 e-10    J\natomic mass unit-kelvin relationship                   1.080 9527 e13        0.000 0019 e13        K\natomic mass unit-kilogram relationship                 1.660 538 782 e-27    0.000 000 083 e-27    kg\natomic unit of 1st hyperpolarizability                 3.206 361 533 e-53    0.000 000 081 e-53    C^3 m^3 J^-2\natomic unit of 2nd hyperpolarizability                 6.235 380 95 e-65     0.000 000 31 e-65     C^4 m^4 J^-3\natomic unit of action                                  1.054 571 628 e-34    0.000 000 053 e-34    J s\natomic unit of charge                                  1.602 176 487 e-19    0.000 000 040 e-19    C\natomic unit of charge density                          1.081 202 300 e12     0.000 000 027 e12     C m^-3\natomic unit of current                                 6.623 617 63 e-3      0.000 000 17 e-3      A\natomic unit of electric dipole mom.                    8.478 352 81 e-30     0.000 000 21 e-30     C m\natomic unit of electric field                          5.142 206 32 e11      0.000 000 13 e11      V m^-1\natomic unit of electric field gradient                 9.717 361 66 e21      0.000 000 24 e21      V m^-2\natomic unit of electric polarizability                 1.648 777 2536 e-41   0.000 000 0034 e-41   C^2 m^2 J^-1\natomic unit of electric potential                      27.211 383 86         0.000 000 68          V\natomic unit of electric quadrupole mom.                4.486 551 07 e-40     0.000 000 11 e-40     C m^2\natomic unit of energy                                  4.359 743 94 e-18     0.000 000 22 e-18     J\natomic unit of force                                   8.238 722 06 e-8      0.000 000 41 e-8      N\natomic unit of length                                  0.529 177 208 59 e-10 0.000 000 000 36 e-10 m\natomic unit of mag. dipole mom.                        1.854 801 830 e-23    0.000 000 046 e-23    J T^-1\natomic unit of mag. flux density                       2.350 517 382 e5      0.000 000 059 e5      T\natomic unit of magnetizability                         7.891 036 433 e-29    0.000 000 027 e-29    J T^-2\natomic unit of mass                                    9.109 382 15 e-31     0.000 000 45 e-31     kg\natomic unit of momentum                                1.992 851 565 e-24    0.000 000 099 e-24    kg m s^-1\natomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1\natomic unit of time                                    2.418 884 326 505 e-17 0.000 000 000 016 e-17 s\natomic unit of velocity                                2.187 691 2541 e6     0.000 000 0015 e6     m s^-1\nAvogadro constant                                      6.022 141 79 e23      0.000 000 30 e23      mol^-1\nBohr magneton                                          927.400 915 e-26      0.000 023 e-26        J T^-1\nBohr magneton in eV/T                                  5.788 381 7555 e-5    0.000 000 0079 e-5    eV T^-1\nBohr magneton in Hz/T                                  13.996 246 04 e9      0.000 000 35 e9       Hz T^-1\nBohr magneton in inverse meters per tesla              46.686 4515           0.000 0012            m^-1 T^-1\nBohr magneton in K/T                                   0.671 7131            0.000 0012            K T^-1\nBohr radius                                            0.529 177 208 59 e-10 0.000 000 000 36 e-10 m\nBoltzmann constant                                     1.380 6504 e-23       0.000 0024 e-23       J K^-1\nBoltzmann constant in eV/K                             8.617 343 e-5         0.000 015 e-5         eV K^-1\nBoltzmann constant in Hz/K                             2.083 6644 e10        0.000 0036 e10        Hz K^-1\nBoltzmann constant in inverse meters per kelvin        69.503 56             0.000 12              m^-1 K^-1\ncharacteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm\nclassical electron radius                              2.817 940 2894 e-15   0.000 000 0058 e-15   m\nCompton wavelength                                     2.426 310 2175 e-12   0.000 000 0033 e-12   m\nCompton wavelength over 2 pi                           386.159 264 59 e-15   0.000 000 53 e-15     m\nconductance quantum                                    7.748 091 7004 e-5    0.000 000 0053 e-5    S\nconventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1\nconventional value of von Klitzing constant            25 812.807            (exact)               ohm\nCu x unit                                              1.002 076 99 e-13     0.000 000 28 e-13     m\ndeuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4\ndeuteron-electron mass ratio                           3670.482 9654         0.000 0016\ndeuteron g factor                                      0.857 438 2308        0.000 000 0072\ndeuteron mag. mom.                                     0.433 073 465 e-26    0.000 000 011 e-26    J T^-1\ndeuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3\ndeuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072\ndeuteron mass                                          3.343 583 20 e-27     0.000 000 17 e-27     kg\ndeuteron mass energy equivalent                        3.005 062 72 e-10     0.000 000 15 e-10     J\ndeuteron mass energy equivalent in MeV                 1875.612 793          0.000 047             MeV\ndeuteron mass in u                                     2.013 553 212 724     0.000 000 000 078     u\ndeuteron molar mass                                    2.013 553 212 724 e-3 0.000 000 000 078 e-3 kg mol^-1\ndeuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11\ndeuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024\ndeuteron-proton mass ratio                             1.999 007 501 08      0.000 000 000 22\ndeuteron rms charge radius                             2.1402 e-15           0.0028 e-15           m\nelectric constant                                      8.854 187 817... e-12 (exact)               F m^-1\nelectron charge to mass quotient                       -1.758 820 150 e11    0.000 000 044 e11     C kg^-1\nelectron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018\nelectron-deuteron mass ratio                           2.724 437 1093 e-4    0.000 000 0012 e-4\nelectron g factor                                      -2.002 319 304 3622   0.000 000 000 0015\nelectron gyromag. ratio                                1.760 859 770 e11     0.000 000 044 e11     s^-1 T^-1\nelectron gyromag. ratio over 2 pi                      28 024.953 64         0.000 70              MHz T^-1\nelectron mag. mom.                                     -928.476 377 e-26     0.000 023 e-26        J T^-1\nelectron mag. mom. anomaly                             1.159 652 181 11 e-3  0.000 000 000 74 e-3\nelectron mag. mom. to Bohr magneton ratio              -1.001 159 652 181 11 0.000 000 000 000 74\nelectron mag. mom. to nuclear magneton ratio           -1838.281 970 92      0.000 000 80\nelectron mass                                          9.109 382 15 e-31     0.000 000 45 e-31     kg\nelectron mass energy equivalent                        8.187 104 38 e-14     0.000 000 41 e-14     J\nelectron mass energy equivalent in MeV                 0.510 998 910         0.000 000 013         MeV\nelectron mass in u                                     5.485 799 0943 e-4    0.000 000 0023 e-4    u\nelectron molar mass                                    5.485 799 0943 e-7    0.000 000 0023 e-7    kg mol^-1\nelectron-muon mag. mom. ratio                          206.766 9877          0.000 0052\nelectron-muon mass ratio                               4.836 331 71 e-3      0.000 000 12 e-3\nelectron-neutron mag. mom. ratio                       960.920 50            0.000 23\nelectron-neutron mass ratio                            5.438 673 4459 e-4    0.000 000 0033 e-4\nelectron-proton mag. mom. ratio                        -658.210 6848         0.000 0054\nelectron-proton mass ratio                             5.446 170 2177 e-4    0.000 000 0024 e-4\nelectron-tau mass ratio                                2.875 64 e-4          0.000 47 e-4\nelectron to alpha particle mass ratio                  1.370 933 555 70 e-4  0.000 000 000 58 e-4\nelectron to shielded helion mag. mom. ratio            864.058 257           0.000 010\nelectron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072\nelectron volt                                          1.602 176 487 e-19    0.000 000 040 e-19    J\nelectron volt-atomic mass unit relationship            1.073 544 188 e-9     0.000 000 027 e-9     u\nelectron volt-hartree relationship                     3.674 932 540 e-2     0.000 000 092 e-2     E_h\nelectron volt-hertz relationship                       2.417 989 454 e14     0.000 000 060 e14     Hz\nelectron volt-inverse meter relationship               8.065 544 65 e5       0.000 000 20 e5       m^-1\nelectron volt-joule relationship                       1.602 176 487 e-19    0.000 000 040 e-19    J\nelectron volt-kelvin relationship                      1.160 4505 e4         0.000 0020 e4         K\nelectron volt-kilogram relationship                    1.782 661 758 e-36    0.000 000 044 e-36    kg\nelementary charge                                      1.602 176 487 e-19    0.000 000 040 e-19    C\nelementary charge over h                               2.417 989 454 e14     0.000 000 060 e14     A J^-1\nFaraday constant                                       96 485.3399           0.0024                C mol^-1\nFaraday constant for conventional electric current     96 485.3401           0.0048                C_90 mol^-1\nFermi coupling constant                                1.166 37 e-5          0.000 01 e-5          GeV^-2\nfine-structure constant                                7.297 352 5376 e-3    0.000 000 0050 e-3\nfirst radiation constant                               3.741 771 18 e-16     0.000 000 19 e-16     W m^2\nfirst radiation constant for spectral radiance         1.191 042 759 e-16    0.000 000 059 e-16    W m^2 sr^-1\nhartree-atomic mass unit relationship                  2.921 262 2986 e-8    0.000 000 0042 e-8    u\nhartree-electron volt relationship                     27.211 383 86         0.000 000 68          eV\nHartree energy                                         4.359 743 94 e-18     0.000 000 22 e-18     J\nHartree energy in eV                                   27.211 383 86         0.000 000 68          eV\nhartree-hertz relationship                             6.579 683 920 722 e15 0.000 000 000 044 e15 Hz\nhartree-inverse meter relationship                     2.194 746 313 705 e7  0.000 000 000 015 e7  m^-1\nhartree-joule relationship                             4.359 743 94 e-18     0.000 000 22 e-18     J\nhartree-kelvin relationship                            3.157 7465 e5         0.000 0055 e5         K\nhartree-kilogram relationship                          4.850 869 34 e-35     0.000 000 24 e-35     kg\nhelion-electron mass ratio                             5495.885 2765         0.000 0052\nhelion mass                                            5.006 411 92 e-27     0.000 000 25 e-27     kg\nhelion mass energy equivalent                          4.499 538 64 e-10     0.000 000 22 e-10     J\nhelion mass energy equivalent in MeV                   2808.391 383          0.000 070             MeV\nhelion mass in u                                       3.014 932 2473        0.000 000 0026        u\nhelion molar mass                                      3.014 932 2473 e-3    0.000 000 0026 e-3    kg mol^-1\nhelion-proton mass ratio                               2.993 152 6713        0.000 000 0026\nhertz-atomic mass unit relationship                    4.439 821 6294 e-24   0.000 000 0064 e-24   u\nhertz-electron volt relationship                       4.135 667 33 e-15     0.000 000 10 e-15     eV\nhertz-hartree relationship                             1.519 829 846 006 e-16 0.000 000 000010e-16 E_h\nhertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1\nhertz-joule relationship                               6.626 068 96 e-34     0.000 000 33 e-34     J\nhertz-kelvin relationship                              4.799 2374 e-11       0.000 0084 e-11       K\nhertz-kilogram relationship                            7.372 496 00 e-51     0.000 000 37 e-51     kg\ninverse fine-structure constant                        137.035 999 679       0.000 000 094\ninverse meter-atomic mass unit relationship            1.331 025 0394 e-15   0.000 000 0019 e-15   u\ninverse meter-electron volt relationship               1.239 841 875 e-6     0.000 000 031 e-6     eV\ninverse meter-hartree relationship                     4.556 335 252 760 e-8 0.000 000 000 030 e-8 E_h\ninverse meter-hertz relationship                       299 792 458           (exact)               Hz\ninverse meter-joule relationship                       1.986 445 501 e-25    0.000 000 099 e-25    J\ninverse meter-kelvin relationship                      1.438 7752 e-2        0.000 0025 e-2        K\ninverse meter-kilogram relationship                    2.210 218 70 e-42     0.000 000 11 e-42     kg\ninverse of conductance quantum                         12 906.403 7787       0.000 0088            ohm\nJosephson constant                                     483 597.891 e9        0.012 e9              Hz V^-1\njoule-atomic mass unit relationship                    6.700 536 41 e9       0.000 000 33 e9       u\njoule-electron volt relationship                       6.241 509 65 e18      0.000 000 16 e18      eV\njoule-hartree relationship                             2.293 712 69 e17      0.000 000 11 e17      E_h\njoule-hertz relationship                               1.509 190 450 e33     0.000 000 075 e33     Hz\njoule-inverse meter relationship                       5.034 117 47 e24      0.000 000 25 e24      m^-1\njoule-kelvin relationship                              7.242 963 e22         0.000 013 e22         K\njoule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg\nkelvin-atomic mass unit relationship                   9.251 098 e-14        0.000 016 e-14        u\nkelvin-electron volt relationship                      8.617 343 e-5         0.000 015 e-5         eV\nkelvin-hartree relationship                            3.166 8153 e-6        0.000 0055 e-6        E_h\nkelvin-hertz relationship                              2.083 6644 e10        0.000 0036 e10        Hz\nkelvin-inverse meter relationship                      69.503 56             0.000 12              m^-1\nkelvin-joule relationship                              1.380 6504 e-23       0.000 0024 e-23       J\nkelvin-kilogram relationship                           1.536 1807 e-40       0.000 0027 e-40       kg\nkilogram-atomic mass unit relationship                 6.022 141 79 e26      0.000 000 30 e26      u\nkilogram-electron volt relationship                    5.609 589 12 e35      0.000 000 14 e35      eV\nkilogram-hartree relationship                          2.061 486 16 e34      0.000 000 10 e34      E_h\nkilogram-hertz relationship                            1.356 392 733 e50     0.000 000 068 e50     Hz\nkilogram-inverse meter relationship                    4.524 439 15 e41      0.000 000 23 e41      m^-1\nkilogram-joule relationship                            8.987 551 787... e16  (exact)               J\nkilogram-kelvin relationship                           6.509 651 e39         0.000 011 e39         K\nlattice parameter of silicon                           543.102 064 e-12      0.000 014 e-12        m\nLoschmidt constant (273.15 K, 101.325 kPa)             2.686 7774 e25        0.000 0047 e25        m^-3\nmag. constant                                          12.566 370 614... e-7 (exact)               N A^-2\nmag. flux quantum                                      2.067 833 667 e-15    0.000 000 052 e-15    Wb\nmolar gas constant                                     8.314 472             0.000 015             J mol^-1 K^-1\nmolar mass constant                                    1 e-3                 (exact)               kg mol^-1\nmolar mass of carbon-12                                12 e-3                (exact)               kg mol^-1\nmolar Planck constant                                  3.990 312 6821 e-10   0.000 000 0057 e-10   J s mol^-1\nmolar Planck constant times c                          0.119 626 564 72      0.000 000 000 17      J m mol^-1\nmolar volume of ideal gas (273.15 K, 100 kPa)          22.710 981 e-3        0.000 040 e-3         m^3 mol^-1\nmolar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 996 e-3        0.000 039 e-3         m^3 mol^-1\nmolar volume of silicon                                12.058 8349 e-6       0.000 0011 e-6        m^3 mol^-1\nMo x unit                                              1.002 099 55 e-13     0.000 000 53 e-13     m\nmuon Compton wavelength                                11.734 441 04 e-15    0.000 000 30 e-15     m\nmuon Compton wavelength over 2 pi                      1.867 594 295 e-15    0.000 000 047 e-15    m\nmuon-electron mass ratio                               206.768 2823          0.000 0052\nmuon g factor                                          -2.002 331 8414       0.000 000 0012\nmuon mag. mom.                                         -4.490 447 86 e-26    0.000 000 16 e-26     J T^-1\nmuon mag. mom. anomaly                                 1.165 920 69 e-3      0.000 000 60 e-3\nmuon mag. mom. to Bohr magneton ratio                  -4.841 970 49 e-3     0.000 000 12 e-3\nmuon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 23\nmuon mass                                              1.883 531 30 e-28     0.000 000 11 e-28     kg\nmuon mass energy equivalent                            1.692 833 510 e-11    0.000 000 095 e-11    J\nmuon mass energy equivalent in MeV                     105.658 3668          0.000 0038            MeV\nmuon mass in u                                         0.113 428 9256        0.000 000 0029        u\nmuon molar mass                                        0.113 428 9256 e-3    0.000 000 0029 e-3    kg mol^-1\nmuon-neutron mass ratio                                0.112 454 5167        0.000 000 0029\nmuon-proton mag. mom. ratio                            -3.183 345 137        0.000 000 085\nmuon-proton mass ratio                                 0.112 609 5261        0.000 000 0029\nmuon-tau mass ratio                                    5.945 92 e-2          0.000 97 e-2\nnatural unit of action                                 1.054 571 628 e-34    0.000 000 053 e-34    J s\nnatural unit of action in eV s                         6.582 118 99 e-16     0.000 000 16 e-16     eV s\nnatural unit of energy                                 8.187 104 38 e-14     0.000 000 41 e-14     J\nnatural unit of energy in MeV                          0.510 998 910         0.000 000 013         MeV\nnatural unit of length                                 386.159 264 59 e-15   0.000 000 53 e-15     m\nnatural unit of mass                                   9.109 382 15 e-31     0.000 000 45 e-31     kg\nnatural unit of momentum                               2.730 924 06 e-22     0.000 000 14 e-22     kg m s^-1\nnatural unit of momentum in MeV/c                      0.510 998 910         0.000 000 013         MeV/c\nnatural unit of time                                   1.288 088 6570 e-21   0.000 000 0018 e-21   s\nnatural unit of velocity                               299 792 458           (exact)               m s^-1\nneutron Compton wavelength                             1.319 590 8951 e-15   0.000 000 0020 e-15   m\nneutron Compton wavelength over 2 pi                   0.210 019 413 82 e-15 0.000 000 000 31 e-15 m\nneutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3\nneutron-electron mass ratio                            1838.683 6605         0.000 0011\nneutron g factor                                       -3.826 085 45         0.000 000 90\nneutron gyromag. ratio                                 1.832 471 85 e8       0.000 000 43 e8       s^-1 T^-1\nneutron gyromag. ratio over 2 pi                       29.164 6954           0.000 0069            MHz T^-1\nneutron mag. mom.                                      -0.966 236 41 e-26    0.000 000 23 e-26     J T^-1\nneutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3\nneutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45\nneutron mass                                           1.674 927 211 e-27    0.000 000 084 e-27    kg\nneutron mass energy equivalent                         1.505 349 505 e-10    0.000 000 075 e-10    J\nneutron mass energy equivalent in MeV                  939.565 346           0.000 023             MeV\nneutron mass in u                                      1.008 664 915 97      0.000 000 000 43      u\nneutron molar mass                                     1.008 664 915 97 e-3  0.000 000 000 43 e-3  kg mol^-1\nneutron-muon mass ratio                                8.892 484 09          0.000 000 23\nneutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16\nneutron-proton mass ratio                              1.001 378 419 18      0.000 000 000 46\nneutron-tau mass ratio                                 0.528 740             0.000 086\nneutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16\nNewtonian constant of gravitation                      6.674 28 e-11         0.000 67 e-11         m^3 kg^-1 s^-2\nNewtonian constant of gravitation over h-bar c         6.708 81 e-39         0.000 67 e-39         (GeV/c^2)^-2\nnuclear magneton                                       5.050 783 24 e-27     0.000 000 13 e-27     J T^-1\nnuclear magneton in eV/T                               3.152 451 2326 e-8    0.000 000 0045 e-8    eV T^-1\nnuclear magneton in inverse meters per tesla           2.542 623 616 e-2     0.000 000 064 e-2     m^-1 T^-1\nnuclear magneton in K/T                                3.658 2637 e-4        0.000 0064 e-4        K T^-1\nnuclear magneton in MHz/T                              7.622 593 84          0.000 000 19          MHz T^-1\nPlanck constant                                        6.626 068 96 e-34     0.000 000 33 e-34     J s\nPlanck constant in eV s                                4.135 667 33 e-15     0.000 000 10 e-15     eV s\nPlanck constant over 2 pi                              1.054 571 628 e-34    0.000 000 053 e-34    J s\nPlanck constant over 2 pi in eV s                      6.582 118 99 e-16     0.000 000 16 e-16     eV s\nPlanck constant over 2 pi times c in MeV fm            197.326 9631          0.000 0049            MeV fm\nPlanck length                                          1.616 252 e-35        0.000 081 e-35        m\nPlanck mass                                            2.176 44 e-8          0.000 11 e-8          kg\nPlanck mass energy equivalent in GeV                   1.220 892 e19         0.000 061 e19         GeV\nPlanck temperature                                     1.416 785 e32         0.000 071 e32         K\nPlanck time                                            5.391 24 e-44         0.000 27 e-44         s\nproton charge to mass quotient                         9.578 833 92 e7       0.000 000 24 e7       C kg^-1\nproton Compton wavelength                              1.321 409 8446 e-15   0.000 000 0019 e-15   m\nproton Compton wavelength over 2 pi                    0.210 308 908 61 e-15 0.000 000 000 30 e-15 m\nproton-electron mass ratio                             1836.152 672 47       0.000 000 80\nproton g factor                                        5.585 694 713         0.000 000 046\nproton gyromag. ratio                                  2.675 222 099 e8      0.000 000 070 e8      s^-1 T^-1\nproton gyromag. ratio over 2 pi                        42.577 4821           0.000 0011            MHz T^-1\nproton mag. mom.                                       1.410 606 662 e-26    0.000 000 037 e-26    J T^-1\nproton mag. mom. to Bohr magneton ratio                1.521 032 209 e-3     0.000 000 012 e-3\nproton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023\nproton mag. shielding correction                       25.694 e-6            0.014 e-6\nproton mass                                            1.672 621 637 e-27    0.000 000 083 e-27    kg\nproton mass energy equivalent                          1.503 277 359 e-10    0.000 000 075 e-10    J\nproton mass energy equivalent in MeV                   938.272 013           0.000 023             MeV\nproton mass in u                                       1.007 276 466 77      0.000 000 000 10      u\nproton molar mass                                      1.007 276 466 77 e-3  0.000 000 000 10 e-3  kg mol^-1\nproton-muon mass ratio                                 8.880 243 39          0.000 000 23\nproton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34\nproton-neutron mass ratio                              0.998 623 478 24      0.000 000 000 46\nproton rms charge radius                               0.8768 e-15           0.0069 e-15           m\nproton-tau mass ratio                                  0.528 012             0.000 086\nquantum of circulation                                 3.636 947 5199 e-4    0.000 000 0050 e-4    m^2 s^-1\nquantum of circulation times 2                         7.273 895 040 e-4     0.000 000 010 e-4     m^2 s^-1\nRydberg constant                                       10 973 731.568 527    0.000 073             m^-1\nRydberg constant times c in Hz                         3.289 841 960 361 e15 0.000 000 000 022 e15 Hz\nRydberg constant times hc in eV                        13.605 691 93         0.000 000 34          eV\nRydberg constant times hc in J                         2.179 871 97 e-18     0.000 000 11 e-18     J\nSackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7047           0.000 0044\nSackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8677           0.000 0044\nsecond radiation constant                              1.438 7752 e-2        0.000 0025 e-2        m K\nshielded helion gyromag. ratio                         2.037 894 730 e8      0.000 000 056 e8      s^-1 T^-1\nshielded helion gyromag. ratio over 2 pi               32.434 101 98         0.000 000 90          MHz T^-1\nshielded helion mag. mom.                              -1.074 552 982 e-26   0.000 000 030 e-26    J T^-1\nshielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3\nshielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025\nshielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011\nshielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033\nshielded proton gyromag. ratio                         2.675 153 362 e8      0.000 000 073 e8      s^-1 T^-1\nshielded proton gyromag. ratio over 2 pi               42.576 3881           0.000 0012            MHz T^-1\nshielded proton mag. mom.                              1.410 570 419 e-26    0.000 000 038 e-26    J T^-1\nshielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3\nshielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030\nspeed of light in vacuum                               299 792 458           (exact)               m s^-1\nstandard acceleration of gravity                       9.806 65              (exact)               m s^-2\nstandard atmosphere                                    101 325               (exact)               Pa\nStefan-Boltzmann constant                              5.670 400 e-8         0.000 040 e-8         W m^-2 K^-4\ntau Compton wavelength                                 0.697 72 e-15         0.000 11 e-15         m\ntau Compton wavelength over 2 pi                       0.111 046 e-15        0.000 018 e-15        m\ntau-electron mass ratio                                3477.48               0.57\ntau mass                                               3.167 77 e-27         0.000 52 e-27         kg\ntau mass energy equivalent                             2.847 05 e-10         0.000 46 e-10         J\ntau mass energy equivalent in MeV                      1776.99               0.29                  MeV\ntau mass in u                                          1.907 68              0.000 31              u\ntau molar mass                                         1.907 68 e-3          0.000 31 e-3          kg mol^-1\ntau-muon mass ratio                                    16.8183               0.0027\ntau-neutron mass ratio                                 1.891 29              0.000 31\ntau-proton mass ratio                                  1.893 90              0.000 31\nThomson cross section                                  0.665 245 8558 e-28   0.000 000 0027 e-28   m^2\ntriton-electron mag. mom. ratio                        -1.620 514 423 e-3    0.000 000 021 e-3\ntriton-electron mass ratio                             5496.921 5269         0.000 0051\ntriton g factor                                        5.957 924 896         0.000 000 076\ntriton mag. mom.                                       1.504 609 361 e-26    0.000 000 042 e-26    J T^-1\ntriton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3\ntriton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038\ntriton mass                                            5.007 355 88 e-27     0.000 000 25 e-27     kg\ntriton mass energy equivalent                          4.500 387 03 e-10     0.000 000 22 e-10     J\ntriton mass energy equivalent in MeV                   2808.920 906          0.000 070             MeV\ntriton mass in u                                       3.015 500 7134        0.000 000 0025        u\ntriton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1\ntriton-neutron mag. mom. ratio                         -1.557 185 53         0.000 000 37\ntriton-proton mag. mom. ratio                          1.066 639 908         0.000 000 010\ntriton-proton mass ratio                               2.993 717 0309        0.000 000 0025\nunified atomic mass unit                               1.660 538 782 e-27    0.000 000 083 e-27    kg\nvon Klitzing constant                                  25 812.807 557        0.000 018             ohm\nweak mixing angle                                      0.222 55              0.000 56\nWien frequency displacement law constant               5.878 933 e10         0.000 010 e10         Hz K^-1\nWien wavelength displacement law constant              2.897 7685 e-3        0.000 0051 e-3        m K')
# Assigning a type to the variable 'txt2006' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'txt2006', str_13475)

# Assigning a Str to a Name (line 468):
str_13476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, (-1)), 'str', '{220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m\nalpha particle-electron mass ratio                     7294.299 5361         0.000 0029\nalpha particle mass                                    6.644 656 75 e-27     0.000 000 29 e-27     kg\nalpha particle mass energy equivalent                  5.971 919 67 e-10     0.000 000 26 e-10     J\nalpha particle mass energy equivalent in MeV           3727.379 240          0.000 082             MeV\nalpha particle mass in u                               4.001 506 179 125     0.000 000 000 062     u\nalpha particle molar mass                              4.001 506 179 125 e-3 0.000 000 000 062 e-3 kg mol^-1\nalpha particle-proton mass ratio                       3.972 599 689 33      0.000 000 000 36\nAngstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m\natomic mass constant                                   1.660 538 921 e-27    0.000 000 073 e-27    kg\natomic mass constant energy equivalent                 1.492 417 954 e-10    0.000 000 066 e-10    J\natomic mass constant energy equivalent in MeV          931.494 061           0.000 021             MeV\natomic mass unit-electron volt relationship            931.494 061 e6        0.000 021 e6          eV\natomic mass unit-hartree relationship                  3.423 177 6845 e7     0.000 000 0024 e7     E_h\natomic mass unit-hertz relationship                    2.252 342 7168 e23    0.000 000 0016 e23    Hz\natomic mass unit-inverse meter relationship            7.513 006 6042 e14    0.000 000 0053 e14    m^-1\natomic mass unit-joule relationship                    1.492 417 954 e-10    0.000 000 066 e-10    J\natomic mass unit-kelvin relationship                   1.080 954 08 e13      0.000 000 98 e13      K\natomic mass unit-kilogram relationship                 1.660 538 921 e-27    0.000 000 073 e-27    kg\natomic unit of 1st hyperpolarizability                 3.206 361 449 e-53    0.000 000 071 e-53    C^3 m^3 J^-2\natomic unit of 2nd hyperpolarizability                 6.235 380 54 e-65     0.000 000 28 e-65     C^4 m^4 J^-3\natomic unit of action                                  1.054 571 726 e-34    0.000 000 047 e-34    J s\natomic unit of charge                                  1.602 176 565 e-19    0.000 000 035 e-19    C\natomic unit of charge density                          1.081 202 338 e12     0.000 000 024 e12     C m^-3\natomic unit of current                                 6.623 617 95 e-3      0.000 000 15 e-3      A\natomic unit of electric dipole mom.                    8.478 353 26 e-30     0.000 000 19 e-30     C m\natomic unit of electric field                          5.142 206 52 e11      0.000 000 11 e11      V m^-1\natomic unit of electric field gradient                 9.717 362 00 e21      0.000 000 21 e21      V m^-2\natomic unit of electric polarizability                 1.648 777 2754 e-41   0.000 000 0016 e-41   C^2 m^2 J^-1\natomic unit of electric potential                      27.211 385 05         0.000 000 60          V\natomic unit of electric quadrupole mom.                4.486 551 331 e-40    0.000 000 099 e-40    C m^2\natomic unit of energy                                  4.359 744 34 e-18     0.000 000 19 e-18     J\natomic unit of force                                   8.238 722 78 e-8      0.000 000 36 e-8      N\natomic unit of length                                  0.529 177 210 92 e-10 0.000 000 000 17 e-10 m\natomic unit of mag. dipole mom.                        1.854 801 936 e-23    0.000 000 041 e-23    J T^-1\natomic unit of mag. flux density                       2.350 517 464 e5      0.000 000 052 e5      T\natomic unit of magnetizability                         7.891 036 607 e-29    0.000 000 013 e-29    J T^-2\natomic unit of mass                                    9.109 382 91 e-31     0.000 000 40 e-31     kg\natomic unit of mom.um                                  1.992 851 740 e-24    0.000 000 088 e-24    kg m s^-1\natomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1\natomic unit of time                                    2.418 884 326 502e-17 0.000 000 000 012e-17 s\natomic unit of velocity                                2.187 691 263 79 e6   0.000 000 000 71 e6   m s^-1\nAvogadro constant                                      6.022 141 29 e23      0.000 000 27 e23      mol^-1\nBohr magneton                                          927.400 968 e-26      0.000 020 e-26        J T^-1\nBohr magneton in eV/T                                  5.788 381 8066 e-5    0.000 000 0038 e-5    eV T^-1\nBohr magneton in Hz/T                                  13.996 245 55 e9      0.000 000 31 e9       Hz T^-1\nBohr magneton in inverse meters per tesla              46.686 4498           0.000 0010            m^-1 T^-1\nBohr magneton in K/T                                   0.671 713 88          0.000 000 61          K T^-1\nBohr radius                                            0.529 177 210 92 e-10 0.000 000 000 17 e-10 m\nBoltzmann constant                                     1.380 6488 e-23       0.000 0013 e-23       J K^-1\nBoltzmann constant in eV/K                             8.617 3324 e-5        0.000 0078 e-5        eV K^-1\nBoltzmann constant in Hz/K                             2.083 6618 e10        0.000 0019 e10        Hz K^-1\nBoltzmann constant in inverse meters per kelvin        69.503 476            0.000 063             m^-1 K^-1\ncharacteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm\nclassical electron radius                              2.817 940 3267 e-15   0.000 000 0027 e-15   m\nCompton wavelength                                     2.426 310 2389 e-12   0.000 000 0016 e-12   m\nCompton wavelength over 2 pi                           386.159 268 00 e-15   0.000 000 25 e-15     m\nconductance quantum                                    7.748 091 7346 e-5    0.000 000 0025 e-5    S\nconventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1\nconventional value of von Klitzing constant            25 812.807            (exact)               ohm\nCu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m\ndeuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4\ndeuteron-electron mass ratio                           3670.482 9652         0.000 0015\ndeuteron g factor                                      0.857 438 2308        0.000 000 0072\ndeuteron mag. mom.                                     0.433 073 489 e-26    0.000 000 010 e-26    J T^-1\ndeuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3\ndeuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072\ndeuteron mass                                          3.343 583 48 e-27     0.000 000 15 e-27     kg\ndeuteron mass energy equivalent                        3.005 062 97 e-10     0.000 000 13 e-10     J\ndeuteron mass energy equivalent in MeV                 1875.612 859          0.000 041             MeV\ndeuteron mass in u                                     2.013 553 212 712     0.000 000 000 077     u\ndeuteron molar mass                                    2.013 553 212 712 e-3 0.000 000 000 077 e-3 kg mol^-1\ndeuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11\ndeuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024\ndeuteron-proton mass ratio                             1.999 007 500 97      0.000 000 000 18\ndeuteron rms charge radius                             2.1424 e-15           0.0021 e-15           m\nelectric constant                                      8.854 187 817... e-12 (exact)               F m^-1\nelectron charge to mass quotient                       -1.758 820 088 e11    0.000 000 039 e11     C kg^-1\nelectron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018\nelectron-deuteron mass ratio                           2.724 437 1095 e-4    0.000 000 0011 e-4\nelectron g factor                                      -2.002 319 304 361 53 0.000 000 000 000 53\nelectron gyromag. ratio                                1.760 859 708 e11     0.000 000 039 e11     s^-1 T^-1\nelectron gyromag. ratio over 2 pi                      28 024.952 66         0.000 62              MHz T^-1\nelectron-helion mass ratio                             1.819 543 0761 e-4    0.000 000 0017 e-4\nelectron mag. mom.                                     -928.476 430 e-26     0.000 021 e-26        J T^-1\nelectron mag. mom. anomaly                             1.159 652 180 76 e-3  0.000 000 000 27 e-3\nelectron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 76 0.000 000 000 000 27\nelectron mag. mom. to nuclear magneton ratio           -1838.281 970 90      0.000 000 75\nelectron mass                                          9.109 382 91 e-31     0.000 000 40 e-31     kg\nelectron mass energy equivalent                        8.187 105 06 e-14     0.000 000 36 e-14     J\nelectron mass energy equivalent in MeV                 0.510 998 928         0.000 000 011         MeV\nelectron mass in u                                     5.485 799 0946 e-4    0.000 000 0022 e-4    u\nelectron molar mass                                    5.485 799 0946 e-7    0.000 000 0022 e-7    kg mol^-1\nelectron-muon mag. mom. ratio                          206.766 9896          0.000 0052\nelectron-muon mass ratio                               4.836 331 66 e-3      0.000 000 12 e-3\nelectron-neutron mag. mom. ratio                       960.920 50            0.000 23\nelectron-neutron mass ratio                            5.438 673 4461 e-4    0.000 000 0032 e-4\nelectron-proton mag. mom. ratio                        -658.210 6848         0.000 0054\nelectron-proton mass ratio                             5.446 170 2178 e-4    0.000 000 0022 e-4\nelectron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4\nelectron to alpha particle mass ratio                  1.370 933 555 78 e-4  0.000 000 000 55 e-4\nelectron to shielded helion mag. mom. ratio            864.058 257           0.000 010\nelectron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072\nelectron-triton mass ratio                             1.819 200 0653 e-4    0.000 000 0017 e-4\nelectron volt                                          1.602 176 565 e-19    0.000 000 035 e-19    J\nelectron volt-atomic mass unit relationship            1.073 544 150 e-9     0.000 000 024 e-9     u\nelectron volt-hartree relationship                     3.674 932 379 e-2     0.000 000 081 e-2     E_h\nelectron volt-hertz relationship                       2.417 989 348 e14     0.000 000 053 e14     Hz\nelectron volt-inverse meter relationship               8.065 544 29 e5       0.000 000 18 e5       m^-1\nelectron volt-joule relationship                       1.602 176 565 e-19    0.000 000 035 e-19    J\nelectron volt-kelvin relationship                      1.160 4519 e4         0.000 0011 e4         K\nelectron volt-kilogram relationship                    1.782 661 845 e-36    0.000 000 039 e-36    kg\nelementary charge                                      1.602 176 565 e-19    0.000 000 035 e-19    C\nelementary charge over h                               2.417 989 348 e14     0.000 000 053 e14     A J^-1\nFaraday constant                                       96 485.3365           0.0021                C mol^-1\nFaraday constant for conventional electric current     96 485.3321           0.0043                C_90 mol^-1\nFermi coupling constant                                1.166 364 e-5         0.000 005 e-5         GeV^-2\nfine-structure constant                                7.297 352 5698 e-3    0.000 000 0024 e-3\nfirst radiation constant                               3.741 771 53 e-16     0.000 000 17 e-16     W m^2\nfirst radiation constant for spectral radiance         1.191 042 869 e-16    0.000 000 053 e-16    W m^2 sr^-1\nhartree-atomic mass unit relationship                  2.921 262 3246 e-8    0.000 000 0021 e-8    u\nhartree-electron volt relationship                     27.211 385 05         0.000 000 60          eV\nHartree energy                                         4.359 744 34 e-18     0.000 000 19 e-18     J\nHartree energy in eV                                   27.211 385 05         0.000 000 60          eV\nhartree-hertz relationship                             6.579 683 920 729 e15 0.000 000 000 033 e15 Hz\nhartree-inverse meter relationship                     2.194 746 313 708 e7  0.000 000 000 011 e7  m^-1\nhartree-joule relationship                             4.359 744 34 e-18     0.000 000 19 e-18     J\nhartree-kelvin relationship                            3.157 7504 e5         0.000 0029 e5         K\nhartree-kilogram relationship                          4.850 869 79 e-35     0.000 000 21 e-35     kg\nhelion-electron mass ratio                             5495.885 2754         0.000 0050\nhelion g factor                                        -4.255 250 613        0.000 000 050\nhelion mag. mom.                                       -1.074 617 486 e-26   0.000 000 027 e-26    J T^-1\nhelion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3\nhelion mag. mom. to nuclear magneton ratio             -2.127 625 306        0.000 000 025\nhelion mass                                            5.006 412 34 e-27     0.000 000 22 e-27     kg\nhelion mass energy equivalent                          4.499 539 02 e-10     0.000 000 20 e-10     J\nhelion mass energy equivalent in MeV                   2808.391 482          0.000 062             MeV\nhelion mass in u                                       3.014 932 2468        0.000 000 0025        u\nhelion molar mass                                      3.014 932 2468 e-3    0.000 000 0025 e-3    kg mol^-1\nhelion-proton mass ratio                               2.993 152 6707        0.000 000 0025\nhertz-atomic mass unit relationship                    4.439 821 6689 e-24   0.000 000 0031 e-24   u\nhertz-electron volt relationship                       4.135 667 516 e-15    0.000 000 091 e-15    eV\nhertz-hartree relationship                             1.519 829 8460045e-16 0.000 000 0000076e-16 E_h\nhertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1\nhertz-joule relationship                               6.626 069 57 e-34     0.000 000 29 e-34     J\nhertz-kelvin relationship                              4.799 2434 e-11       0.000 0044 e-11       K\nhertz-kilogram relationship                            7.372 496 68 e-51     0.000 000 33 e-51     kg\ninverse fine-structure constant                        137.035 999 074       0.000 000 044\ninverse meter-atomic mass unit relationship            1.331 025 051 20 e-15 0.000 000 000 94 e-15 u\ninverse meter-electron volt relationship               1.239 841 930 e-6     0.000 000 027 e-6     eV\ninverse meter-hartree relationship                     4.556 335 252 755 e-8 0.000 000 000 023 e-8 E_h\ninverse meter-hertz relationship                       299 792 458           (exact)               Hz\ninverse meter-joule relationship                       1.986 445 684 e-25    0.000 000 088 e-25    J\ninverse meter-kelvin relationship                      1.438 7770 e-2        0.000 0013 e-2        K\ninverse meter-kilogram relationship                    2.210 218 902 e-42    0.000 000 098 e-42    kg\ninverse of conductance quantum                         12 906.403 7217       0.000 0042            ohm\nJosephson constant                                     483 597.870 e9        0.011 e9              Hz V^-1\njoule-atomic mass unit relationship                    6.700 535 85 e9       0.000 000 30 e9       u\njoule-electron volt relationship                       6.241 509 34 e18      0.000 000 14 e18      eV\njoule-hartree relationship                             2.293 712 48 e17      0.000 000 10 e17      E_h\njoule-hertz relationship                               1.509 190 311 e33     0.000 000 067 e33     Hz\njoule-inverse meter relationship                       5.034 117 01 e24      0.000 000 22 e24      m^-1\njoule-kelvin relationship                              7.242 9716 e22        0.000 0066 e22        K\njoule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg\nkelvin-atomic mass unit relationship                   9.251 0868 e-14       0.000 0084 e-14       u\nkelvin-electron volt relationship                      8.617 3324 e-5        0.000 0078 e-5        eV\nkelvin-hartree relationship                            3.166 8114 e-6        0.000 0029 e-6        E_h\nkelvin-hertz relationship                              2.083 6618 e10        0.000 0019 e10        Hz\nkelvin-inverse meter relationship                      69.503 476            0.000 063             m^-1\nkelvin-joule relationship                              1.380 6488 e-23       0.000 0013 e-23       J\nkelvin-kilogram relationship                           1.536 1790 e-40       0.000 0014 e-40       kg\nkilogram-atomic mass unit relationship                 6.022 141 29 e26      0.000 000 27 e26      u\nkilogram-electron volt relationship                    5.609 588 85 e35      0.000 000 12 e35      eV\nkilogram-hartree relationship                          2.061 485 968 e34     0.000 000 091 e34     E_h\nkilogram-hertz relationship                            1.356 392 608 e50     0.000 000 060 e50     Hz\nkilogram-inverse meter relationship                    4.524 438 73 e41      0.000 000 20 e41      m^-1\nkilogram-joule relationship                            8.987 551 787... e16  (exact)               J\nkilogram-kelvin relationship                           6.509 6582 e39        0.000 0059 e39        K\nlattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m\nLoschmidt constant (273.15 K, 100 kPa)                 2.651 6462 e25        0.000 0024 e25        m^-3\nLoschmidt constant (273.15 K, 101.325 kPa)             2.686 7805 e25        0.000 0024 e25        m^-3\nmag. constant                                          12.566 370 614... e-7 (exact)               N A^-2\nmag. flux quantum                                      2.067 833 758 e-15    0.000 000 046 e-15    Wb\nmolar gas constant                                     8.314 4621            0.000 0075            J mol^-1 K^-1\nmolar mass constant                                    1 e-3                 (exact)               kg mol^-1\nmolar mass of carbon-12                                12 e-3                (exact)               kg mol^-1\nmolar Planck constant                                  3.990 312 7176 e-10   0.000 000 0028 e-10   J s mol^-1\nmolar Planck constant times c                          0.119 626 565 779     0.000 000 000 084     J m mol^-1\nmolar volume of ideal gas (273.15 K, 100 kPa)          22.710 953 e-3        0.000 021 e-3         m^3 mol^-1\nmolar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 968 e-3        0.000 020 e-3         m^3 mol^-1\nmolar volume of silicon                                12.058 833 01 e-6     0.000 000 80 e-6      m^3 mol^-1\nMo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m\nmuon Compton wavelength                                11.734 441 03 e-15    0.000 000 30 e-15     m\nmuon Compton wavelength over 2 pi                      1.867 594 294 e-15    0.000 000 047 e-15    m\nmuon-electron mass ratio                               206.768 2843          0.000 0052\nmuon g factor                                          -2.002 331 8418       0.000 000 0013\nmuon mag. mom.                                         -4.490 448 07 e-26    0.000 000 15 e-26     J T^-1\nmuon mag. mom. anomaly                                 1.165 920 91 e-3      0.000 000 63 e-3\nmuon mag. mom. to Bohr magneton ratio                  -4.841 970 44 e-3     0.000 000 12 e-3\nmuon mag. mom. to nuclear magneton ratio               -8.890 596 97         0.000 000 22\nmuon mass                                              1.883 531 475 e-28    0.000 000 096 e-28    kg\nmuon mass energy equivalent                            1.692 833 667 e-11    0.000 000 086 e-11    J\nmuon mass energy equivalent in MeV                     105.658 3715          0.000 0035            MeV\nmuon mass in u                                         0.113 428 9267        0.000 000 0029        u\nmuon molar mass                                        0.113 428 9267 e-3    0.000 000 0029 e-3    kg mol^-1\nmuon-neutron mass ratio                                0.112 454 5177        0.000 000 0028\nmuon-proton mag. mom. ratio                            -3.183 345 107        0.000 000 084\nmuon-proton mass ratio                                 0.112 609 5272        0.000 000 0028\nmuon-tau mass ratio                                    5.946 49 e-2          0.000 54 e-2\nnatural unit of action                                 1.054 571 726 e-34    0.000 000 047 e-34    J s\nnatural unit of action in eV s                         6.582 119 28 e-16     0.000 000 15 e-16     eV s\nnatural unit of energy                                 8.187 105 06 e-14     0.000 000 36 e-14     J\nnatural unit of energy in MeV                          0.510 998 928         0.000 000 011         MeV\nnatural unit of length                                 386.159 268 00 e-15   0.000 000 25 e-15     m\nnatural unit of mass                                   9.109 382 91 e-31     0.000 000 40 e-31     kg\nnatural unit of mom.um                                 2.730 924 29 e-22     0.000 000 12 e-22     kg m s^-1\nnatural unit of mom.um in MeV/c                        0.510 998 928         0.000 000 011         MeV/c\nnatural unit of time                                   1.288 088 668 33 e-21 0.000 000 000 83 e-21 s\nnatural unit of velocity                               299 792 458           (exact)               m s^-1\nneutron Compton wavelength                             1.319 590 9068 e-15   0.000 000 0011 e-15   m\nneutron Compton wavelength over 2 pi                   0.210 019 415 68 e-15 0.000 000 000 17 e-15 m\nneutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3\nneutron-electron mass ratio                            1838.683 6605         0.000 0011\nneutron g factor                                       -3.826 085 45         0.000 000 90\nneutron gyromag. ratio                                 1.832 471 79 e8       0.000 000 43 e8       s^-1 T^-1\nneutron gyromag. ratio over 2 pi                       29.164 6943           0.000 0069            MHz T^-1\nneutron mag. mom.                                      -0.966 236 47 e-26    0.000 000 23 e-26     J T^-1\nneutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3\nneutron mag. mom. to nuclear magneton ratio            -1.913 042 72         0.000 000 45\nneutron mass                                           1.674 927 351 e-27    0.000 000 074 e-27    kg\nneutron mass energy equivalent                         1.505 349 631 e-10    0.000 000 066 e-10    J\nneutron mass energy equivalent in MeV                  939.565 379           0.000 021             MeV\nneutron mass in u                                      1.008 664 916 00      0.000 000 000 43      u\nneutron molar mass                                     1.008 664 916 00 e-3  0.000 000 000 43 e-3  kg mol^-1\nneutron-muon mass ratio                                8.892 484 00          0.000 000 22\nneutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16\nneutron-proton mass difference                         2.305 573 92 e-30     0.000 000 76 e-30\nneutron-proton mass difference energy equivalent       2.072 146 50 e-13     0.000 000 68 e-13\nneutron-proton mass difference energy equivalent in MeV 1.293 332 17          0.000 000 42\nneutron-proton mass difference in u                    0.001 388 449 19      0.000 000 000 45\nneutron-proton mass ratio                              1.001 378 419 17      0.000 000 000 45\nneutron-tau mass ratio                                 0.528 790             0.000 048\nneutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16\nNewtonian constant of gravitation                      6.673 84 e-11         0.000 80 e-11         m^3 kg^-1 s^-2\nNewtonian constant of gravitation over h-bar c         6.708 37 e-39         0.000 80 e-39         (GeV/c^2)^-2\nnuclear magneton                                       5.050 783 53 e-27     0.000 000 11 e-27     J T^-1\nnuclear magneton in eV/T                               3.152 451 2605 e-8    0.000 000 0022 e-8    eV T^-1\nnuclear magneton in inverse meters per tesla           2.542 623 527 e-2     0.000 000 056 e-2     m^-1 T^-1\nnuclear magneton in K/T                                3.658 2682 e-4        0.000 0033 e-4        K T^-1\nnuclear magneton in MHz/T                              7.622 593 57          0.000 000 17          MHz T^-1\nPlanck constant                                        6.626 069 57 e-34     0.000 000 29 e-34     J s\nPlanck constant in eV s                                4.135 667 516 e-15    0.000 000 091 e-15    eV s\nPlanck constant over 2 pi                              1.054 571 726 e-34    0.000 000 047 e-34    J s\nPlanck constant over 2 pi in eV s                      6.582 119 28 e-16     0.000 000 15 e-16     eV s\nPlanck constant over 2 pi times c in MeV fm            197.326 9718          0.000 0044            MeV fm\nPlanck length                                          1.616 199 e-35        0.000 097 e-35        m\nPlanck mass                                            2.176 51 e-8          0.000 13 e-8          kg\nPlanck mass energy equivalent in GeV                   1.220 932 e19         0.000 073 e19         GeV\nPlanck temperature                                     1.416 833 e32         0.000 085 e32         K\nPlanck time                                            5.391 06 e-44         0.000 32 e-44         s\nproton charge to mass quotient                         9.578 833 58 e7       0.000 000 21 e7       C kg^-1\nproton Compton wavelength                              1.321 409 856 23 e-15 0.000 000 000 94 e-15 m\nproton Compton wavelength over 2 pi                    0.210 308 910 47 e-15 0.000 000 000 15 e-15 m\nproton-electron mass ratio                             1836.152 672 45       0.000 000 75\nproton g factor                                        5.585 694 713         0.000 000 046\nproton gyromag. ratio                                  2.675 222 005 e8      0.000 000 063 e8      s^-1 T^-1\nproton gyromag. ratio over 2 pi                        42.577 4806           0.000 0010            MHz T^-1\nproton mag. mom.                                       1.410 606 743 e-26    0.000 000 033 e-26    J T^-1\nproton mag. mom. to Bohr magneton ratio                1.521 032 210 e-3     0.000 000 012 e-3\nproton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023\nproton mag. shielding correction                       25.694 e-6            0.014 e-6\nproton mass                                            1.672 621 777 e-27    0.000 000 074 e-27    kg\nproton mass energy equivalent                          1.503 277 484 e-10    0.000 000 066 e-10    J\nproton mass energy equivalent in MeV                   938.272 046           0.000 021             MeV\nproton mass in u                                       1.007 276 466 812     0.000 000 000 090     u\nproton molar mass                                      1.007 276 466 812 e-3 0.000 000 000 090 e-3 kg mol^-1\nproton-muon mass ratio                                 8.880 243 31          0.000 000 22\nproton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34\nproton-neutron mass ratio                              0.998 623 478 26      0.000 000 000 45\nproton rms charge radius                               0.8775 e-15           0.0051 e-15           m\nproton-tau mass ratio                                  0.528 063             0.000 048\nquantum of circulation                                 3.636 947 5520 e-4    0.000 000 0024 e-4    m^2 s^-1\nquantum of circulation times 2                         7.273 895 1040 e-4    0.000 000 0047 e-4    m^2 s^-1\nRydberg constant                                       10 973 731.568 539    0.000 055             m^-1\nRydberg constant times c in Hz                         3.289 841 960 364 e15 0.000 000 000 017 e15 Hz\nRydberg constant times hc in eV                        13.605 692 53         0.000 000 30          eV\nRydberg constant times hc in J                         2.179 872 171 e-18    0.000 000 096 e-18    J\nSackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7078           0.000 0023\nSackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8708           0.000 0023\nsecond radiation constant                              1.438 7770 e-2        0.000 0013 e-2        m K\nshielded helion gyromag. ratio                         2.037 894 659 e8      0.000 000 051 e8      s^-1 T^-1\nshielded helion gyromag. ratio over 2 pi               32.434 100 84         0.000 000 81          MHz T^-1\nshielded helion mag. mom.                              -1.074 553 044 e-26   0.000 000 027 e-26    J T^-1\nshielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3\nshielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025\nshielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011\nshielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033\nshielded proton gyromag. ratio                         2.675 153 268 e8      0.000 000 066 e8      s^-1 T^-1\nshielded proton gyromag. ratio over 2 pi               42.576 3866           0.000 0010            MHz T^-1\nshielded proton mag. mom.                              1.410 570 499 e-26    0.000 000 035 e-26    J T^-1\nshielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3\nshielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030\nspeed of light in vacuum                               299 792 458           (exact)               m s^-1\nstandard acceleration of gravity                       9.806 65              (exact)               m s^-2\nstandard atmosphere                                    101 325               (exact)               Pa\nstandard-state pressure                                100 000               (exact)               Pa\nStefan-Boltzmann constant                              5.670 373 e-8         0.000 021 e-8         W m^-2 K^-4\ntau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m\ntau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m\ntau-electron mass ratio                                3477.15               0.31\ntau mass                                               3.167 47 e-27         0.000 29 e-27         kg\ntau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J\ntau mass energy equivalent in MeV                      1776.82               0.16                  MeV\ntau mass in u                                          1.907 49              0.000 17              u\ntau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1\ntau-muon mass ratio                                    16.8167               0.0015\ntau-neutron mass ratio                                 1.891 11              0.000 17\ntau-proton mass ratio                                  1.893 72              0.000 17\nThomson cross section                                  0.665 245 8734 e-28   0.000 000 0013 e-28   m^2\ntriton-electron mass ratio                             5496.921 5267         0.000 0050\ntriton g factor                                        5.957 924 896         0.000 000 076\ntriton mag. mom.                                       1.504 609 447 e-26    0.000 000 038 e-26    J T^-1\ntriton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3\ntriton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038\ntriton mass                                            5.007 356 30 e-27     0.000 000 22 e-27     kg\ntriton mass energy equivalent                          4.500 387 41 e-10     0.000 000 20 e-10     J\ntriton mass energy equivalent in MeV                   2808.921 005          0.000 062             MeV\ntriton mass in u                                       3.015 500 7134        0.000 000 0025        u\ntriton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1\ntriton-proton mass ratio                               2.993 717 0308        0.000 000 0025\nunified atomic mass unit                               1.660 538 921 e-27    0.000 000 073 e-27    kg\nvon Klitzing constant                                  25 812.807 4434       0.000 0084            ohm\nweak mixing angle                                      0.2223                0.0021\nWien frequency displacement law constant               5.878 9254 e10        0.000 0053 e10        Hz K^-1\nWien wavelength displacement law constant              2.897 7721 e-3        0.000 0026 e-3        m K')
# Assigning a type to the variable 'txt2010' (line 468)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 0), 'txt2010', str_13476)

# Assigning a Str to a Name (line 805):
str_13477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, (-1)), 'str', '{220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m\nalpha particle-electron mass ratio                     7294.299 541 36       0.000 000 24\nalpha particle mass                                    6.644 657 230 e-27    0.000 000 082 e-27    kg\nalpha particle mass energy equivalent                  5.971 920 097 e-10    0.000 000 073 e-10    J\nalpha particle mass energy equivalent in MeV           3727.379 378          0.000 023             MeV\nalpha particle mass in u                               4.001 506 179 127     0.000 000 000 063     u\nalpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 063 e-3 kg mol^-1\nalpha particle-proton mass ratio                       3.972 599 689 07      0.000 000 000 36\nAngstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m\natomic mass constant                                   1.660 539 040 e-27    0.000 000 020 e-27    kg\natomic mass constant energy equivalent                 1.492 418 062 e-10    0.000 000 018 e-10    J\natomic mass constant energy equivalent in MeV          931.494 0954          0.000 0057            MeV\natomic mass unit-electron volt relationship            931.494 0954 e6       0.000 0057 e6         eV\natomic mass unit-hartree relationship                  3.423 177 6902 e7     0.000 000 0016 e7     E_h\natomic mass unit-hertz relationship                    2.252 342 7206 e23    0.000 000 0010 e23    Hz\natomic mass unit-inverse meter relationship            7.513 006 6166 e14    0.000 000 0034 e14    m^-1\natomic mass unit-joule relationship                    1.492 418 062 e-10    0.000 000 018 e-10    J\natomic mass unit-kelvin relationship                   1.080 954 38 e13      0.000 000 62 e13      K\natomic mass unit-kilogram relationship                 1.660 539 040 e-27    0.000 000 020 e-27    kg\natomic unit of 1st hyperpolarizability                 3.206 361 329 e-53    0.000 000 020 e-53    C^3 m^3 J^-2\natomic unit of 2nd hyperpolarizability                 6.235 380 085 e-65    0.000 000 077 e-65    C^4 m^4 J^-3\natomic unit of action                                  1.054 571 800 e-34    0.000 000 013 e-34    J s\natomic unit of charge                                  1.602 176 6208 e-19   0.000 000 0098 e-19   C\natomic unit of charge density                          1.081 202 3770 e12    0.000 000 0067 e12    C m^-3\natomic unit of current                                 6.623 618 183 e-3     0.000 000 041 e-3     A\natomic unit of electric dipole mom.                    8.478 353 552 e-30    0.000 000 052 e-30    C m\natomic unit of electric field                          5.142 206 707 e11     0.000 000 032 e11     V m^-1\natomic unit of electric field gradient                 9.717 362 356 e21     0.000 000 060 e21     V m^-2\natomic unit of electric polarizability                 1.648 777 2731 e-41   0.000 000 0011 e-41   C^2 m^2 J^-1\natomic unit of electric potential                      27.211 386 02         0.000 000 17          V\natomic unit of electric quadrupole mom.                4.486 551 484 e-40    0.000 000 028 e-40    C m^2\natomic unit of energy                                  4.359 744 650 e-18    0.000 000 054 e-18    J\natomic unit of force                                   8.238 723 36 e-8      0.000 000 10 e-8      N\natomic unit of length                                  0.529 177 210 67 e-10 0.000 000 000 12 e-10 m\natomic unit of mag. dipole mom.                        1.854 801 999 e-23    0.000 000 011 e-23    J T^-1\natomic unit of mag. flux density                       2.350 517 550 e5      0.000 000 014 e5      T\natomic unit of magnetizability                         7.891 036 5886 e-29   0.000 000 0090 e-29   J T^-2\natomic unit of mass                                    9.109 383 56 e-31     0.000 000 11 e-31     kg\natomic unit of mom.um                                  1.992 851 882 e-24    0.000 000 024 e-24    kg m s^-1\natomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1\natomic unit of time                                    2.418 884 326509e-17  0.000 000 000014e-17  s\natomic unit of velocity                                2.187 691 262 77 e6   0.000 000 000 50 e6   m s^-1\nAvogadro constant                                      6.022 140 857 e23     0.000 000 074 e23     mol^-1\nBohr magneton                                          927.400 9994 e-26     0.000 0057 e-26       J T^-1\nBohr magneton in eV/T                                  5.788 381 8012 e-5    0.000 000 0026 e-5    eV T^-1\nBohr magneton in Hz/T                                  13.996 245 042 e9     0.000 000 086 e9      Hz T^-1\nBohr magneton in inverse meters per tesla              46.686 448 14         0.000 000 29          m^-1 T^-1\nBohr magneton in K/T                                   0.671 714 05          0.000 000 39          K T^-1\nBohr radius                                            0.529 177 210 67 e-10 0.000 000 000 12 e-10 m\nBoltzmann constant                                     1.380 648 52 e-23     0.000 000 79 e-23     J K^-1\nBoltzmann constant in eV/K                             8.617 3303 e-5        0.000 0050 e-5        eV K^-1\nBoltzmann constant in Hz/K                             2.083 6612 e10        0.000 0012 e10        Hz K^-1\nBoltzmann constant in inverse meters per kelvin        69.503 457            0.000 040             m^-1 K^-1\ncharacteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm\nclassical electron radius                              2.817 940 3227 e-15   0.000 000 0019 e-15   m\nCompton wavelength                                     2.426 310 2367 e-12   0.000 000 0011 e-12   m\nCompton wavelength over 2 pi                           386.159 267 64 e-15   0.000 000 18 e-15     m\nconductance quantum                                    7.748 091 7310 e-5    0.000 000 0018 e-5    S\nconventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1\nconventional value of von Klitzing constant            25 812.807            (exact)               ohm\nCu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m\ndeuteron-electron mag. mom. ratio                      -4.664 345 535 e-4    0.000 000 026 e-4\ndeuteron-electron mass ratio                           3670.482 967 85       0.000 000 13\ndeuteron g factor                                      0.857 438 2311        0.000 000 0048\ndeuteron mag. mom.                                     0.433 073 5040 e-26   0.000 000 0036 e-26   J T^-1\ndeuteron mag. mom. to Bohr magneton ratio              0.466 975 4554 e-3    0.000 000 0026 e-3\ndeuteron mag. mom. to nuclear magneton ratio           0.857 438 2311        0.000 000 0048\ndeuteron mass                                          3.343 583 719 e-27    0.000 000 041 e-27    kg\ndeuteron mass energy equivalent                        3.005 063 183 e-10    0.000 000 037 e-10    J\ndeuteron mass energy equivalent in MeV                 1875.612 928          0.000 012             MeV\ndeuteron mass in u                                     2.013 553 212 745     0.000 000 000 040     u\ndeuteron molar mass                                    2.013 553 212 745 e-3 0.000 000 000 040 e-3 kg mol^-1\ndeuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11\ndeuteron-proton mag. mom. ratio                        0.307 012 2077        0.000 000 0015\ndeuteron-proton mass ratio                             1.999 007 500 87      0.000 000 000 19\ndeuteron rms charge radius                             2.1413 e-15           0.0025 e-15           m\nelectric constant                                      8.854 187 817... e-12 (exact)               F m^-1\nelectron charge to mass quotient                       -1.758 820 024 e11    0.000 000 011 e11     C kg^-1\nelectron-deuteron mag. mom. ratio                      -2143.923 499         0.000 012\nelectron-deuteron mass ratio                           2.724 437 107 484 e-4 0.000 000 000 096 e-4\nelectron g factor                                      -2.002 319 304 361 82 0.000 000 000 000 52\nelectron gyromag. ratio                                1.760 859 644 e11     0.000 000 011 e11     s^-1 T^-1\nelectron gyromag. ratio over 2 pi                      28 024.951 64         0.000 17              MHz T^-1\nelectron-helion mass ratio                             1.819 543 074 854 e-4 0.000 000 000 088 e-4\nelectron mag. mom.                                     -928.476 4620 e-26    0.000 0057 e-26       J T^-1\nelectron mag. mom. anomaly                             1.159 652 180 91 e-3  0.000 000 000 26 e-3\nelectron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 91 0.000 000 000 000 26\nelectron mag. mom. to nuclear magneton ratio           -1838.281 972 34      0.000 000 17\nelectron mass                                          9.109 383 56 e-31     0.000 000 11 e-31     kg\nelectron mass energy equivalent                        8.187 105 65 e-14     0.000 000 10 e-14     J\nelectron mass energy equivalent in MeV                 0.510 998 9461        0.000 000 0031        MeV\nelectron mass in u                                     5.485 799 090 70 e-4  0.000 000 000 16 e-4  u\nelectron molar mass                                    5.485 799 090 70 e-7  0.000 000 000 16 e-7  kg mol^-1\nelectron-muon mag. mom. ratio                          206.766 9880          0.000 0046\nelectron-muon mass ratio                               4.836 331 70 e-3      0.000 000 11 e-3\nelectron-neutron mag. mom. ratio                       960.920 50            0.000 23\nelectron-neutron mass ratio                            5.438 673 4428 e-4    0.000 000 0027 e-4\nelectron-proton mag. mom. ratio                        -658.210 6866         0.000 0020\nelectron-proton mass ratio                             5.446 170 213 52 e-4  0.000 000 000 52 e-4\nelectron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4\nelectron to alpha particle mass ratio                  1.370 933 554 798 e-4 0.000 000 000 045 e-4\nelectron to shielded helion mag. mom. ratio            864.058 257           0.000 010\nelectron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072\nelectron-triton mass ratio                             1.819 200 062 203 e-4 0.000 000 000 084 e-4\nelectron volt                                          1.602 176 6208 e-19   0.000 000 0098 e-19   J\nelectron volt-atomic mass unit relationship            1.073 544 1105 e-9    0.000 000 0066 e-9    u\nelectron volt-hartree relationship                     3.674 932 248 e-2     0.000 000 023 e-2     E_h\nelectron volt-hertz relationship                       2.417 989 262 e14     0.000 000 015 e14     Hz\nelectron volt-inverse meter relationship               8.065 544 005 e5      0.000 000 050 e5      m^-1\nelectron volt-joule relationship                       1.602 176 6208 e-19   0.000 000 0098 e-19   J\nelectron volt-kelvin relationship                      1.160 452 21 e4       0.000 000 67 e4       K\nelectron volt-kilogram relationship                    1.782 661 907 e-36    0.000 000 011 e-36    kg\nelementary charge                                      1.602 176 6208 e-19   0.000 000 0098 e-19   C\nelementary charge over h                               2.417 989 262 e14     0.000 000 015 e14     A J^-1\nFaraday constant                                       96 485.332 89         0.000 59              C mol^-1\nFaraday constant for conventional electric current     96 485.3251           0.0012                C_90 mol^-1\nFermi coupling constant                                1.166 3787 e-5        0.000 0006 e-5        GeV^-2\nfine-structure constant                                7.297 352 5664 e-3    0.000 000 0017 e-3\nfirst radiation constant                               3.741 771 790 e-16    0.000 000 046 e-16    W m^2\nfirst radiation constant for spectral radiance         1.191 042 953 e-16    0.000 000 015 e-16    W m^2 sr^-1\nhartree-atomic mass unit relationship                  2.921 262 3197 e-8    0.000 000 0013 e-8    u\nhartree-electron volt relationship                     27.211 386 02         0.000 000 17          eV\nHartree energy                                         4.359 744 650 e-18    0.000 000 054 e-18    J\nHartree energy in eV                                   27.211 386 02         0.000 000 17          eV\nhartree-hertz relationship                             6.579 683 920 711 e15 0.000 000 000 039 e15 Hz\nhartree-inverse meter relationship                     2.194 746 313 702 e7  0.000 000 000 013 e7  m^-1\nhartree-joule relationship                             4.359 744 650 e-18    0.000 000 054 e-18    J\nhartree-kelvin relationship                            3.157 7513 e5         0.000 0018 e5         K\nhartree-kilogram relationship                          4.850 870 129 e-35    0.000 000 060 e-35    kg\nhelion-electron mass ratio                             5495.885 279 22       0.000 000 27\nhelion g factor                                        -4.255 250 616        0.000 000 050\nhelion mag. mom.                                       -1.074 617 522 e-26   0.000 000 014 e-26    J T^-1\nhelion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3\nhelion mag. mom. to nuclear magneton ratio             -2.127 625 308        0.000 000 025\nhelion mass                                            5.006 412 700 e-27    0.000 000 062 e-27    kg\nhelion mass energy equivalent                          4.499 539 341 e-10    0.000 000 055 e-10    J\nhelion mass energy equivalent in MeV                   2808.391 586          0.000 017             MeV\nhelion mass in u                                       3.014 932 246 73      0.000 000 000 12      u\nhelion molar mass                                      3.014 932 246 73 e-3  0.000 000 000 12 e-3  kg mol^-1\nhelion-proton mass ratio                               2.993 152 670 46      0.000 000 000 29\nhertz-atomic mass unit relationship                    4.439 821 6616 e-24   0.000 000 0020 e-24   u\nhertz-electron volt relationship                       4.135 667 662 e-15    0.000 000 025 e-15    eV\nhertz-hartree relationship                             1.5198298460088 e-16  0.0000000000090e-16   E_h\nhertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1\nhertz-joule relationship                               6.626 070 040 e-34    0.000 000 081 e-34    J\nhertz-kelvin relationship                              4.799 2447 e-11       0.000 0028 e-11       K\nhertz-kilogram relationship                            7.372 497 201 e-51    0.000 000 091 e-51    kg\ninverse fine-structure constant                        137.035 999 139       0.000 000 031\ninverse meter-atomic mass unit relationship            1.331 025 049 00 e-15 0.000 000 000 61 e-15 u\ninverse meter-electron volt relationship               1.239 841 9739 e-6    0.000 000 0076 e-6    eV\ninverse meter-hartree relationship                     4.556 335 252 767 e-8 0.000 000 000 027 e-8 E_h\ninverse meter-hertz relationship                       299 792 458           (exact)               Hz\ninverse meter-joule relationship                       1.986 445 824 e-25    0.000 000 024 e-25    J\ninverse meter-kelvin relationship                      1.438 777 36 e-2      0.000 000 83 e-2      K\ninverse meter-kilogram relationship                    2.210 219 057 e-42    0.000 000 027 e-42    kg\ninverse of conductance quantum                         12 906.403 7278       0.000 0029            ohm\nJosephson constant                                     483 597.8525 e9       0.0030 e9             Hz V^-1\njoule-atomic mass unit relationship                    6.700 535 363 e9      0.000 000 082 e9      u\njoule-electron volt relationship                       6.241 509 126 e18     0.000 000 038 e18     eV\njoule-hartree relationship                             2.293 712 317 e17     0.000 000 028 e17     E_h\njoule-hertz relationship                               1.509 190 205 e33     0.000 000 019 e33     Hz\njoule-inverse meter relationship                       5.034 116 651 e24     0.000 000 062 e24     m^-1\njoule-kelvin relationship                              7.242 9731 e22        0.000 0042 e22        K\njoule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg\nkelvin-atomic mass unit relationship                   9.251 0842 e-14       0.000 0053 e-14       u\nkelvin-electron volt relationship                      8.617 3303 e-5        0.000 0050 e-5        eV\nkelvin-hartree relationship                            3.166 8105 e-6        0.000 0018 e-6        E_h\nkelvin-hertz relationship                              2.083 6612 e10        0.000 0012 e10        Hz\nkelvin-inverse meter relationship                      69.503 457            0.000 040             m^-1\nkelvin-joule relationship                              1.380 648 52 e-23     0.000 000 79 e-23     J\nkelvin-kilogram relationship                           1.536 178 65 e-40     0.000 000 88 e-40     kg\nkilogram-atomic mass unit relationship                 6.022 140 857 e26     0.000 000 074 e26     u\nkilogram-electron volt relationship                    5.609 588 650 e35     0.000 000 034 e35     eV\nkilogram-hartree relationship                          2.061 485 823 e34     0.000 000 025 e34     E_h\nkilogram-hertz relationship                            1.356 392 512 e50     0.000 000 017 e50     Hz\nkilogram-inverse meter relationship                    4.524 438 411 e41     0.000 000 056 e41     m^-1\nkilogram-joule relationship                            8.987 551 787... e16  (exact)               J\nkilogram-kelvin relationship                           6.509 6595 e39        0.000 0037 e39        K\nlattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m\nLoschmidt constant (273.15 K, 100 kPa)                 2.651 6467 e25        0.000 0015 e25        m^-3\nLoschmidt constant (273.15 K, 101.325 kPa)             2.686 7811 e25        0.000 0015 e25        m^-3\nmag. constant                                          12.566 370 614... e-7 (exact)               N A^-2\nmag. flux quantum                                      2.067 833 831 e-15    0.000 000 013 e-15    Wb\nmolar gas constant                                     8.314 4598            0.000 0048            J mol^-1 K^-1\nmolar mass constant                                    1 e-3                 (exact)               kg mol^-1\nmolar mass of carbon-12                                12 e-3                (exact)               kg mol^-1\nmolar Planck constant                                  3.990 312 7110 e-10   0.000 000 0018 e-10   J s mol^-1\nmolar Planck constant times c                          0.119 626 565 582     0.000 000 000 054     J m mol^-1\nmolar volume of ideal gas (273.15 K, 100 kPa)          22.710 947 e-3        0.000 013 e-3         m^3 mol^-1\nmolar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 962 e-3        0.000 013 e-3         m^3 mol^-1\nmolar volume of silicon                                12.058 832 14 e-6     0.000 000 61 e-6      m^3 mol^-1\nMo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m\nmuon Compton wavelength                                11.734 441 11 e-15    0.000 000 26 e-15     m\nmuon Compton wavelength over 2 pi                      1.867 594 308 e-15    0.000 000 042 e-15    m\nmuon-electron mass ratio                               206.768 2826          0.000 0046\nmuon g factor                                          -2.002 331 8418       0.000 000 0013\nmuon mag. mom.                                         -4.490 448 26 e-26    0.000 000 10 e-26     J T^-1\nmuon mag. mom. anomaly                                 1.165 920 89 e-3      0.000 000 63 e-3\nmuon mag. mom. to Bohr magneton ratio                  -4.841 970 48 e-3     0.000 000 11 e-3\nmuon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 20\nmuon mass                                              1.883 531 594 e-28    0.000 000 048 e-28    kg\nmuon mass energy equivalent                            1.692 833 774 e-11    0.000 000 043 e-11    J\nmuon mass energy equivalent in MeV                     105.658 3745          0.000 0024            MeV\nmuon mass in u                                         0.113 428 9257        0.000 000 0025        u\nmuon molar mass                                        0.113 428 9257 e-3    0.000 000 0025 e-3    kg mol^-1\nmuon-neutron mass ratio                                0.112 454 5167        0.000 000 0025\nmuon-proton mag. mom. ratio                            -3.183 345 142        0.000 000 071\nmuon-proton mass ratio                                 0.112 609 5262        0.000 000 0025\nmuon-tau mass ratio                                    5.946 49 e-2          0.000 54 e-2\nnatural unit of action                                 1.054 571 800 e-34    0.000 000 013 e-34    J s\nnatural unit of action in eV s                         6.582 119 514 e-16    0.000 000 040 e-16    eV s\nnatural unit of energy                                 8.187 105 65 e-14     0.000 000 10 e-14     J\nnatural unit of energy in MeV                          0.510 998 9461        0.000 000 0031        MeV\nnatural unit of length                                 386.159 267 64 e-15   0.000 000 18 e-15     m\nnatural unit of mass                                   9.109 383 56 e-31     0.000 000 11 e-31     kg\nnatural unit of mom.um                                 2.730 924 488 e-22    0.000 000 034 e-22    kg m s^-1\nnatural unit of mom.um in MeV/c                        0.510 998 9461        0.000 000 0031        MeV/c\nnatural unit of time                                   1.288 088 667 12 e-21 0.000 000 000 58 e-21 s\nnatural unit of velocity                               299 792 458           (exact)               m s^-1\nneutron Compton wavelength                             1.319 590 904 81 e-15 0.000 000 000 88 e-15 m\nneutron Compton wavelength over 2 pi                   0.210 019 415 36 e-15 0.000 000 000 14 e-15 m\nneutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3\nneutron-electron mass ratio                            1838.683 661 58       0.000 000 90\nneutron g factor                                       -3.826 085 45         0.000 000 90\nneutron gyromag. ratio                                 1.832 471 72 e8       0.000 000 43 e8       s^-1 T^-1\nneutron gyromag. ratio over 2 pi                       29.164 6933           0.000 0069            MHz T^-1\nneutron mag. mom.                                      -0.966 236 50 e-26    0.000 000 23 e-26     J T^-1\nneutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3\nneutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45\nneutron mass                                           1.674 927 471 e-27    0.000 000 021 e-27    kg\nneutron mass energy equivalent                         1.505 349 739 e-10    0.000 000 019 e-10    J\nneutron mass energy equivalent in MeV                  939.565 4133          0.000 0058            MeV\nneutron mass in u                                      1.008 664 915 88      0.000 000 000 49      u\nneutron molar mass                                     1.008 664 915 88 e-3  0.000 000 000 49 e-3  kg mol^-1\nneutron-muon mass ratio                                8.892 484 08          0.000 000 20\nneutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16\nneutron-proton mass difference                         2.305 573 77 e-30     0.000 000 85 e-30\nneutron-proton mass difference energy equivalent       2.072 146 37 e-13     0.000 000 76 e-13\nneutron-proton mass difference energy equivalent in MeV 1.293 332 05         0.000 000 48\nneutron-proton mass difference in u                    0.001 388 449 00      0.000 000 000 51\nneutron-proton mass ratio                              1.001 378 418 98      0.000 000 000 51\nneutron-tau mass ratio                                 0.528 790             0.000 048\nneutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16\nNewtonian constant of gravitation                      6.674 08 e-11         0.000 31 e-11         m^3 kg^-1 s^-2\nNewtonian constant of gravitation over h-bar c         6.708 61 e-39         0.000 31 e-39         (GeV/c^2)^-2\nnuclear magneton                                       5.050 783 699 e-27    0.000 000 031 e-27    J T^-1\nnuclear magneton in eV/T                               3.152 451 2550 e-8    0.000 000 0015 e-8    eV T^-1\nnuclear magneton in inverse meters per tesla           2.542 623 432 e-2     0.000 000 016 e-2     m^-1 T^-1\nnuclear magneton in K/T                                3.658 2690 e-4        0.000 0021 e-4        K T^-1\nnuclear magneton in MHz/T                              7.622 593 285         0.000 000 047         MHz T^-1\nPlanck constant                                        6.626 070 040 e-34    0.000 000 081 e-34    J s\nPlanck constant in eV s                                4.135 667 662 e-15    0.000 000 025 e-15    eV s\nPlanck constant over 2 pi                              1.054 571 800 e-34    0.000 000 013 e-34    J s\nPlanck constant over 2 pi in eV s                      6.582 119 514 e-16    0.000 000 040 e-16    eV s\nPlanck constant over 2 pi times c in MeV fm            197.326 9788          0.000 0012            MeV fm\nPlanck length                                          1.616 229 e-35        0.000 038 e-35        m\nPlanck mass                                            2.176 470 e-8         0.000 051 e-8         kg\nPlanck mass energy equivalent in GeV                   1.220 910 e19         0.000 029 e19         GeV\nPlanck temperature                                     1.416 808 e32         0.000 033 e32         K\nPlanck time                                            5.391 16 e-44         0.000 13 e-44         s\nproton charge to mass quotient                         9.578 833 226 e7      0.000 000 059 e7      C kg^-1\nproton Compton wavelength                              1.321 409 853 96 e-15 0.000 000 000 61 e-15 m\nproton Compton wavelength over 2 pi                    0.210 308910109e-15   0.000 000 000097e-15  m\nproton-electron mass ratio                             1836.152 673 89       0.000 000 17\nproton g factor                                        5.585 694 702         0.000 000 017\nproton gyromag. ratio                                  2.675 221 900 e8      0.000 000 018 e8      s^-1 T^-1\nproton gyromag. ratio over 2 pi                        42.577 478 92         0.000 000 29          MHz T^-1\nproton mag. mom.                                       1.410 606 7873 e-26   0.000 000 0097 e-26   J T^-1\nproton mag. mom. to Bohr magneton ratio                1.521 032 2053 e-3    0.000 000 0046 e-3\nproton mag. mom. to nuclear magneton ratio             2.792 847 3508        0.000 000 0085\nproton mag. shielding correction                       25.691 e-6            0.011 e-6\nproton mass                                            1.672 621 898 e-27    0.000 000 021 e-27    kg\nproton mass energy equivalent                          1.503 277 593 e-10    0.000 000 018 e-10    J\nproton mass energy equivalent in MeV                   938.272 0813          0.000 0058            MeV\nproton mass in u                                       1.007 276 466 879     0.000 000 000 091     u\nproton molar mass                                      1.007 276 466 879 e-3 0.000 000 000 091 e-3 kg mol^-1\nproton-muon mass ratio                                 8.880 243 38          0.000 000 20\nproton-neutron mag. mom. ratio                         -1.459 898 05         0.000 000 34\nproton-neutron mass ratio                              0.998 623 478 44      0.000 000 000 51\nproton rms charge radius                               0.8751 e-15           0.0061 e-15           m\nproton-tau mass ratio                                  0.528 063             0.000 048\nquantum of circulation                                 3.636 947 5486 e-4    0.000 000 0017 e-4    m^2 s^-1\nquantum of circulation times 2                         7.273 895 0972 e-4    0.000 000 0033 e-4    m^2 s^-1\nRydberg constant                                       10 973 731.568 508    0.000 065             m^-1\nRydberg constant times c in Hz                         3.289 841 960 355 e15 0.000 000 000 019 e15 Hz\nRydberg constant times hc in eV                        13.605 693 009        0.000 000 084         eV\nRydberg constant times hc in J                         2.179 872 325 e-18    0.000 000 027 e-18    J\nSackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7084           0.000 0014\nSackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8714           0.000 0014\nsecond radiation constant                              1.438 777 36 e-2      0.000 000 83 e-2      m K\nshielded helion gyromag. ratio                         2.037 894 585 e8      0.000 000 027 e8      s^-1 T^-1\nshielded helion gyromag. ratio over 2 pi               32.434 099 66         0.000 000 43          MHz T^-1\nshielded helion mag. mom.                              -1.074 553 080 e-26   0.000 000 014 e-26    J T^-1\nshielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3\nshielded helion mag. mom. to nuclear magneton ratio    -2.127 497 720        0.000 000 025\nshielded helion to proton mag. mom. ratio              -0.761 766 5603       0.000 000 0092\nshielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033\nshielded proton gyromag. ratio                         2.675 153 171 e8      0.000 000 033 e8      s^-1 T^-1\nshielded proton gyromag. ratio over 2 pi               42.576 385 07         0.000 000 53          MHz T^-1\nshielded proton mag. mom.                              1.410 570 547 e-26    0.000 000 018 e-26    J T^-1\nshielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3\nshielded proton mag. mom. to nuclear magneton ratio    2.792 775 600         0.000 000 030\nspeed of light in vacuum                               299 792 458           (exact)               m s^-1\nstandard acceleration of gravity                       9.806 65              (exact)               m s^-2\nstandard atmosphere                                    101 325               (exact)               Pa\nstandard-state pressure                                100 000               (exact)               Pa\nStefan-Boltzmann constant                              5.670 367 e-8         0.000 013 e-8         W m^-2 K^-4\ntau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m\ntau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m\ntau-electron mass ratio                                3477.15               0.31\ntau mass                                               3.167 47 e-27         0.000 29 e-27         kg\ntau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J\ntau mass energy equivalent in MeV                      1776.82               0.16                  MeV\ntau mass in u                                          1.907 49              0.000 17              u\ntau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1\ntau-muon mass ratio                                    16.8167               0.0015\ntau-neutron mass ratio                                 1.891 11              0.000 17\ntau-proton mass ratio                                  1.893 72              0.000 17\nThomson cross section                                  0.665 245 871 58 e-28 0.000 000 000 91 e-28 m^2\ntriton-electron mass ratio                             5496.921 535 88       0.000 000 26\ntriton g factor                                        5.957 924 920         0.000 000 028\ntriton mag. mom.                                       1.504 609 503 e-26    0.000 000 012 e-26    J T^-1\ntriton mag. mom. to Bohr magneton ratio                1.622 393 6616 e-3    0.000 000 0076 e-3\ntriton mag. mom. to nuclear magneton ratio             2.978 962 460         0.000 000 014\ntriton mass                                            5.007 356 665 e-27    0.000 000 062 e-27    kg\ntriton mass energy equivalent                          4.500 387 735 e-10    0.000 000 055 e-10    J\ntriton mass energy equivalent in MeV                   2808.921 112          0.000 017             MeV\ntriton mass in u                                       3.015 500 716 32      0.000 000 000 11      u\ntriton molar mass                                      3.015 500 716 32 e-3  0.000 000 000 11 e-3  kg mol^-1\ntriton-proton mass ratio                               2.993 717 033 48      0.000 000 000 22\nunified atomic mass unit                               1.660 539 040 e-27    0.000 000 020 e-27    kg\nvon Klitzing constant                                  25 812.807 4555       0.000 0059            ohm\nweak mixing angle                                      0.2223                0.0021\nWien frequency displacement law constant               5.878 9238 e10        0.000 0034 e10        Hz K^-1\nWien wavelength displacement law constant              2.897 7729 e-3        0.000 0017 e-3        m K')
# Assigning a type to the variable 'txt2014' (line 805)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 0), 'txt2014', str_13477)

# Assigning a Dict to a Name (line 1144):

# Obtaining an instance of the builtin type 'dict' (line 1144)
dict_13478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1144)

# Assigning a type to the variable 'physical_constants' (line 1144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 0), 'physical_constants', dict_13478)

@norecursion
def parse_constants(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_constants'
    module_type_store = module_type_store.open_function_context('parse_constants', 1147, 0, False)
    
    # Passed parameters checking function
    parse_constants.stypy_localization = localization
    parse_constants.stypy_type_of_self = None
    parse_constants.stypy_type_store = module_type_store
    parse_constants.stypy_function_name = 'parse_constants'
    parse_constants.stypy_param_names_list = ['d']
    parse_constants.stypy_varargs_param_name = None
    parse_constants.stypy_kwargs_param_name = None
    parse_constants.stypy_call_defaults = defaults
    parse_constants.stypy_call_varargs = varargs
    parse_constants.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_constants', ['d'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_constants', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_constants(...)' code ##################

    
    # Assigning a Dict to a Name (line 1148):
    
    # Obtaining an instance of the builtin type 'dict' (line 1148)
    dict_13479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1148)
    
    # Assigning a type to the variable 'constants' (line 1148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 4), 'constants', dict_13479)
    
    
    # Call to split(...): (line 1149)
    # Processing the call arguments (line 1149)
    str_13482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1149, 24), 'str', '\n')
    # Processing the call keyword arguments (line 1149)
    kwargs_13483 = {}
    # Getting the type of 'd' (line 1149)
    d_13480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 16), 'd', False)
    # Obtaining the member 'split' of a type (line 1149)
    split_13481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 16), d_13480, 'split')
    # Calling split(args, kwargs) (line 1149)
    split_call_result_13484 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 16), split_13481, *[str_13482], **kwargs_13483)
    
    # Testing the type of a for loop iterable (line 1149)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1149, 4), split_call_result_13484)
    # Getting the type of the for loop variable (line 1149)
    for_loop_var_13485 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1149, 4), split_call_result_13484)
    # Assigning a type to the variable 'line' (line 1149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'line', for_loop_var_13485)
    # SSA begins for a for statement (line 1149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1150):
    
    # Call to rstrip(...): (line 1150)
    # Processing the call keyword arguments (line 1150)
    kwargs_13492 = {}
    
    # Obtaining the type of the subscript
    int_13486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1150, 21), 'int')
    slice_13487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1150, 15), None, int_13486, None)
    # Getting the type of 'line' (line 1150)
    line_13488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 15), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 1150)
    getitem___13489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1150, 15), line_13488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1150)
    subscript_call_result_13490 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 15), getitem___13489, slice_13487)
    
    # Obtaining the member 'rstrip' of a type (line 1150)
    rstrip_13491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1150, 15), subscript_call_result_13490, 'rstrip')
    # Calling rstrip(args, kwargs) (line 1150)
    rstrip_call_result_13493 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 15), rstrip_13491, *[], **kwargs_13492)
    
    # Assigning a type to the variable 'name' (line 1150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 8), 'name', rstrip_call_result_13493)
    
    # Assigning a Call to a Name (line 1151):
    
    # Call to replace(...): (line 1151)
    # Processing the call arguments (line 1151)
    str_13506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 51), 'str', '...')
    str_13507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 58), 'str', '')
    # Processing the call keyword arguments (line 1151)
    kwargs_13508 = {}
    
    # Call to replace(...): (line 1151)
    # Processing the call arguments (line 1151)
    str_13501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 34), 'str', ' ')
    str_13502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 39), 'str', '')
    # Processing the call keyword arguments (line 1151)
    kwargs_13503 = {}
    
    # Obtaining the type of the subscript
    int_13494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 19), 'int')
    int_13495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 22), 'int')
    slice_13496 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1151, 14), int_13494, int_13495, None)
    # Getting the type of 'line' (line 1151)
    line_13497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 14), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 1151)
    getitem___13498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 14), line_13497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1151)
    subscript_call_result_13499 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 14), getitem___13498, slice_13496)
    
    # Obtaining the member 'replace' of a type (line 1151)
    replace_13500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 14), subscript_call_result_13499, 'replace')
    # Calling replace(args, kwargs) (line 1151)
    replace_call_result_13504 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 14), replace_13500, *[str_13501, str_13502], **kwargs_13503)
    
    # Obtaining the member 'replace' of a type (line 1151)
    replace_13505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 14), replace_call_result_13504, 'replace')
    # Calling replace(args, kwargs) (line 1151)
    replace_call_result_13509 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 14), replace_13505, *[str_13506, str_13507], **kwargs_13508)
    
    # Assigning a type to the variable 'val' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 8), 'val', replace_call_result_13509)
    
    # Assigning a Call to a Name (line 1152):
    
    # Call to float(...): (line 1152)
    # Processing the call arguments (line 1152)
    # Getting the type of 'val' (line 1152)
    val_13511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 20), 'val', False)
    # Processing the call keyword arguments (line 1152)
    kwargs_13512 = {}
    # Getting the type of 'float' (line 1152)
    float_13510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 14), 'float', False)
    # Calling float(args, kwargs) (line 1152)
    float_call_result_13513 = invoke(stypy.reporting.localization.Localization(__file__, 1152, 14), float_13510, *[val_13511], **kwargs_13512)
    
    # Assigning a type to the variable 'val' (line 1152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 8), 'val', float_call_result_13513)
    
    # Assigning a Call to a Name (line 1153):
    
    # Call to replace(...): (line 1153)
    # Processing the call arguments (line 1153)
    str_13526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 54), 'str', '(exact)')
    str_13527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 65), 'str', '0')
    # Processing the call keyword arguments (line 1153)
    kwargs_13528 = {}
    
    # Call to replace(...): (line 1153)
    # Processing the call arguments (line 1153)
    str_13521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 37), 'str', ' ')
    str_13522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 42), 'str', '')
    # Processing the call keyword arguments (line 1153)
    kwargs_13523 = {}
    
    # Obtaining the type of the subscript
    int_13514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 22), 'int')
    int_13515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 25), 'int')
    slice_13516 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1153, 17), int_13514, int_13515, None)
    # Getting the type of 'line' (line 1153)
    line_13517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 17), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 1153)
    getitem___13518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 17), line_13517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1153)
    subscript_call_result_13519 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 17), getitem___13518, slice_13516)
    
    # Obtaining the member 'replace' of a type (line 1153)
    replace_13520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 17), subscript_call_result_13519, 'replace')
    # Calling replace(args, kwargs) (line 1153)
    replace_call_result_13524 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 17), replace_13520, *[str_13521, str_13522], **kwargs_13523)
    
    # Obtaining the member 'replace' of a type (line 1153)
    replace_13525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 17), replace_call_result_13524, 'replace')
    # Calling replace(args, kwargs) (line 1153)
    replace_call_result_13529 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 17), replace_13525, *[str_13526, str_13527], **kwargs_13528)
    
    # Assigning a type to the variable 'uncert' (line 1153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 8), 'uncert', replace_call_result_13529)
    
    # Assigning a Call to a Name (line 1154):
    
    # Call to float(...): (line 1154)
    # Processing the call arguments (line 1154)
    # Getting the type of 'uncert' (line 1154)
    uncert_13531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 23), 'uncert', False)
    # Processing the call keyword arguments (line 1154)
    kwargs_13532 = {}
    # Getting the type of 'float' (line 1154)
    float_13530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 17), 'float', False)
    # Calling float(args, kwargs) (line 1154)
    float_call_result_13533 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 17), float_13530, *[uncert_13531], **kwargs_13532)
    
    # Assigning a type to the variable 'uncert' (line 1154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 8), 'uncert', float_call_result_13533)
    
    # Assigning a Call to a Name (line 1155):
    
    # Call to rstrip(...): (line 1155)
    # Processing the call keyword arguments (line 1155)
    kwargs_13540 = {}
    
    # Obtaining the type of the subscript
    int_13534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1155, 21), 'int')
    slice_13535 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1155, 16), int_13534, None, None)
    # Getting the type of 'line' (line 1155)
    line_13536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 16), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 1155)
    getitem___13537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 16), line_13536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1155)
    subscript_call_result_13538 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 16), getitem___13537, slice_13535)
    
    # Obtaining the member 'rstrip' of a type (line 1155)
    rstrip_13539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 16), subscript_call_result_13538, 'rstrip')
    # Calling rstrip(args, kwargs) (line 1155)
    rstrip_call_result_13541 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 16), rstrip_13539, *[], **kwargs_13540)
    
    # Assigning a type to the variable 'units' (line 1155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 8), 'units', rstrip_call_result_13541)
    
    # Assigning a Tuple to a Subscript (line 1156):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1156)
    tuple_13542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1156, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1156)
    # Adding element type (line 1156)
    # Getting the type of 'val' (line 1156)
    val_13543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 27), 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1156, 27), tuple_13542, val_13543)
    # Adding element type (line 1156)
    # Getting the type of 'units' (line 1156)
    units_13544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 32), 'units')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1156, 27), tuple_13542, units_13544)
    # Adding element type (line 1156)
    # Getting the type of 'uncert' (line 1156)
    uncert_13545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 39), 'uncert')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1156, 27), tuple_13542, uncert_13545)
    
    # Getting the type of 'constants' (line 1156)
    constants_13546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 8), 'constants')
    # Getting the type of 'name' (line 1156)
    name_13547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 18), 'name')
    # Storing an element on a container (line 1156)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1156, 8), constants_13546, (name_13547, tuple_13542))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'constants' (line 1157)
    constants_13548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 11), 'constants')
    # Assigning a type to the variable 'stypy_return_type' (line 1157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 4), 'stypy_return_type', constants_13548)
    
    # ################# End of 'parse_constants(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_constants' in the type store
    # Getting the type of 'stypy_return_type' (line 1147)
    stypy_return_type_13549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13549)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_constants'
    return stypy_return_type_13549

# Assigning a type to the variable 'parse_constants' (line 1147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 0), 'parse_constants', parse_constants)

# Assigning a Call to a Name (line 1160):

# Call to parse_constants(...): (line 1160)
# Processing the call arguments (line 1160)
# Getting the type of 'txt2002' (line 1160)
txt2002_13551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 43), 'txt2002', False)
# Processing the call keyword arguments (line 1160)
kwargs_13552 = {}
# Getting the type of 'parse_constants' (line 1160)
parse_constants_13550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 27), 'parse_constants', False)
# Calling parse_constants(args, kwargs) (line 1160)
parse_constants_call_result_13553 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 27), parse_constants_13550, *[txt2002_13551], **kwargs_13552)

# Assigning a type to the variable '_physical_constants_2002' (line 1160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1160, 0), '_physical_constants_2002', parse_constants_call_result_13553)

# Assigning a Call to a Name (line 1161):

# Call to parse_constants(...): (line 1161)
# Processing the call arguments (line 1161)
# Getting the type of 'txt2006' (line 1161)
txt2006_13555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 43), 'txt2006', False)
# Processing the call keyword arguments (line 1161)
kwargs_13556 = {}
# Getting the type of 'parse_constants' (line 1161)
parse_constants_13554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 27), 'parse_constants', False)
# Calling parse_constants(args, kwargs) (line 1161)
parse_constants_call_result_13557 = invoke(stypy.reporting.localization.Localization(__file__, 1161, 27), parse_constants_13554, *[txt2006_13555], **kwargs_13556)

# Assigning a type to the variable '_physical_constants_2006' (line 1161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 0), '_physical_constants_2006', parse_constants_call_result_13557)

# Assigning a Call to a Name (line 1162):

# Call to parse_constants(...): (line 1162)
# Processing the call arguments (line 1162)
# Getting the type of 'txt2010' (line 1162)
txt2010_13559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 43), 'txt2010', False)
# Processing the call keyword arguments (line 1162)
kwargs_13560 = {}
# Getting the type of 'parse_constants' (line 1162)
parse_constants_13558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 27), 'parse_constants', False)
# Calling parse_constants(args, kwargs) (line 1162)
parse_constants_call_result_13561 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 27), parse_constants_13558, *[txt2010_13559], **kwargs_13560)

# Assigning a type to the variable '_physical_constants_2010' (line 1162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 0), '_physical_constants_2010', parse_constants_call_result_13561)

# Assigning a Call to a Name (line 1163):

# Call to parse_constants(...): (line 1163)
# Processing the call arguments (line 1163)
# Getting the type of 'txt2014' (line 1163)
txt2014_13563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 43), 'txt2014', False)
# Processing the call keyword arguments (line 1163)
kwargs_13564 = {}
# Getting the type of 'parse_constants' (line 1163)
parse_constants_13562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 27), 'parse_constants', False)
# Calling parse_constants(args, kwargs) (line 1163)
parse_constants_call_result_13565 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 27), parse_constants_13562, *[txt2014_13563], **kwargs_13564)

# Assigning a type to the variable '_physical_constants_2014' (line 1163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1163, 0), '_physical_constants_2014', parse_constants_call_result_13565)

# Call to update(...): (line 1166)
# Processing the call arguments (line 1166)
# Getting the type of '_physical_constants_2002' (line 1166)
_physical_constants_2002_13568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 26), '_physical_constants_2002', False)
# Processing the call keyword arguments (line 1166)
kwargs_13569 = {}
# Getting the type of 'physical_constants' (line 1166)
physical_constants_13566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 0), 'physical_constants', False)
# Obtaining the member 'update' of a type (line 1166)
update_13567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 0), physical_constants_13566, 'update')
# Calling update(args, kwargs) (line 1166)
update_call_result_13570 = invoke(stypy.reporting.localization.Localization(__file__, 1166, 0), update_13567, *[_physical_constants_2002_13568], **kwargs_13569)


# Call to update(...): (line 1167)
# Processing the call arguments (line 1167)
# Getting the type of '_physical_constants_2006' (line 1167)
_physical_constants_2006_13573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 26), '_physical_constants_2006', False)
# Processing the call keyword arguments (line 1167)
kwargs_13574 = {}
# Getting the type of 'physical_constants' (line 1167)
physical_constants_13571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 0), 'physical_constants', False)
# Obtaining the member 'update' of a type (line 1167)
update_13572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 0), physical_constants_13571, 'update')
# Calling update(args, kwargs) (line 1167)
update_call_result_13575 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 0), update_13572, *[_physical_constants_2006_13573], **kwargs_13574)


# Call to update(...): (line 1168)
# Processing the call arguments (line 1168)
# Getting the type of '_physical_constants_2010' (line 1168)
_physical_constants_2010_13578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 26), '_physical_constants_2010', False)
# Processing the call keyword arguments (line 1168)
kwargs_13579 = {}
# Getting the type of 'physical_constants' (line 1168)
physical_constants_13576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 0), 'physical_constants', False)
# Obtaining the member 'update' of a type (line 1168)
update_13577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1168, 0), physical_constants_13576, 'update')
# Calling update(args, kwargs) (line 1168)
update_call_result_13580 = invoke(stypy.reporting.localization.Localization(__file__, 1168, 0), update_13577, *[_physical_constants_2010_13578], **kwargs_13579)


# Call to update(...): (line 1169)
# Processing the call arguments (line 1169)
# Getting the type of '_physical_constants_2014' (line 1169)
_physical_constants_2014_13583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 26), '_physical_constants_2014', False)
# Processing the call keyword arguments (line 1169)
kwargs_13584 = {}
# Getting the type of 'physical_constants' (line 1169)
physical_constants_13581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 0), 'physical_constants', False)
# Obtaining the member 'update' of a type (line 1169)
update_13582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 0), physical_constants_13581, 'update')
# Calling update(args, kwargs) (line 1169)
update_call_result_13585 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 0), update_13582, *[_physical_constants_2014_13583], **kwargs_13584)


# Assigning a Name to a Name (line 1170):
# Getting the type of '_physical_constants_2014' (line 1170)
_physical_constants_2014_13586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 21), '_physical_constants_2014')
# Assigning a type to the variable '_current_constants' (line 1170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 0), '_current_constants', _physical_constants_2014_13586)

# Assigning a Str to a Name (line 1171):
str_13587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 18), 'str', 'CODATA 2014')
# Assigning a type to the variable '_current_codata' (line 1171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 0), '_current_codata', str_13587)

# Assigning a Dict to a Name (line 1174):

# Obtaining an instance of the builtin type 'dict' (line 1174)
dict_13588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1174, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1174)

# Assigning a type to the variable '_obsolete_constants' (line 1174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 0), '_obsolete_constants', dict_13588)

# Getting the type of 'physical_constants' (line 1175)
physical_constants_13589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 9), 'physical_constants')
# Testing the type of a for loop iterable (line 1175)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1175, 0), physical_constants_13589)
# Getting the type of the for loop variable (line 1175)
for_loop_var_13590 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1175, 0), physical_constants_13589)
# Assigning a type to the variable 'k' (line 1175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1175, 0), 'k', for_loop_var_13590)
# SSA begins for a for statement (line 1175)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Getting the type of 'k' (line 1176)
k_13591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 7), 'k')
# Getting the type of '_current_constants' (line 1176)
_current_constants_13592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 16), '_current_constants')
# Applying the binary operator 'notin' (line 1176)
result_contains_13593 = python_operator(stypy.reporting.localization.Localization(__file__, 1176, 7), 'notin', k_13591, _current_constants_13592)

# Testing the type of an if condition (line 1176)
if_condition_13594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1176, 4), result_contains_13593)
# Assigning a type to the variable 'if_condition_13594' (line 1176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 4), 'if_condition_13594', if_condition_13594)
# SSA begins for if statement (line 1176)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Subscript (line 1177):
# Getting the type of 'True' (line 1177)
True_13595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 33), 'True')
# Getting the type of '_obsolete_constants' (line 1177)
_obsolete_constants_13596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 8), '_obsolete_constants')
# Getting the type of 'k' (line 1177)
k_13597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 28), 'k')
# Storing an element on a container (line 1177)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1177, 8), _obsolete_constants_13596, (k_13597, True_13595))
# SSA join for if statement (line 1176)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 1180):

# Obtaining an instance of the builtin type 'dict' (line 1180)
dict_13598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1180)

# Assigning a type to the variable '_aliases' (line 1180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 0), '_aliases', dict_13598)

# Getting the type of '_physical_constants_2002' (line 1181)
_physical_constants_2002_13599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 9), '_physical_constants_2002')
# Testing the type of a for loop iterable (line 1181)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1181, 0), _physical_constants_2002_13599)
# Getting the type of the for loop variable (line 1181)
for_loop_var_13600 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1181, 0), _physical_constants_2002_13599)
# Assigning a type to the variable 'k' (line 1181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 0), 'k', for_loop_var_13600)
# SSA begins for a for statement (line 1181)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


str_13601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 7), 'str', 'magn.')
# Getting the type of 'k' (line 1182)
k_13602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 18), 'k')
# Applying the binary operator 'in' (line 1182)
result_contains_13603 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 7), 'in', str_13601, k_13602)

# Testing the type of an if condition (line 1182)
if_condition_13604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1182, 4), result_contains_13603)
# Assigning a type to the variable 'if_condition_13604' (line 1182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 4), 'if_condition_13604', if_condition_13604)
# SSA begins for if statement (line 1182)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Subscript (line 1183):

# Call to replace(...): (line 1183)
# Processing the call arguments (line 1183)
str_13607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 32), 'str', 'magn.')
str_13608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 41), 'str', 'mag.')
# Processing the call keyword arguments (line 1183)
kwargs_13609 = {}
# Getting the type of 'k' (line 1183)
k_13605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 22), 'k', False)
# Obtaining the member 'replace' of a type (line 1183)
replace_13606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1183, 22), k_13605, 'replace')
# Calling replace(args, kwargs) (line 1183)
replace_call_result_13610 = invoke(stypy.reporting.localization.Localization(__file__, 1183, 22), replace_13606, *[str_13607, str_13608], **kwargs_13609)

# Getting the type of '_aliases' (line 1183)
_aliases_13611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 8), '_aliases')
# Getting the type of 'k' (line 1183)
k_13612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 17), 'k')
# Storing an element on a container (line 1183)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1183, 8), _aliases_13611, (k_13612, replace_call_result_13610))
# SSA join for if statement (line 1182)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Getting the type of '_physical_constants_2006' (line 1184)
_physical_constants_2006_13613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 9), '_physical_constants_2006')
# Testing the type of a for loop iterable (line 1184)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1184, 0), _physical_constants_2006_13613)
# Getting the type of the for loop variable (line 1184)
for_loop_var_13614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1184, 0), _physical_constants_2006_13613)
# Assigning a type to the variable 'k' (line 1184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 0), 'k', for_loop_var_13614)
# SSA begins for a for statement (line 1184)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


str_13615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 7), 'str', 'momentum')
# Getting the type of 'k' (line 1185)
k_13616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 21), 'k')
# Applying the binary operator 'in' (line 1185)
result_contains_13617 = python_operator(stypy.reporting.localization.Localization(__file__, 1185, 7), 'in', str_13615, k_13616)

# Testing the type of an if condition (line 1185)
if_condition_13618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1185, 4), result_contains_13617)
# Assigning a type to the variable 'if_condition_13618' (line 1185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'if_condition_13618', if_condition_13618)
# SSA begins for if statement (line 1185)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Subscript (line 1186):

# Call to replace(...): (line 1186)
# Processing the call arguments (line 1186)
str_13621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 32), 'str', 'momentum')
str_13622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 44), 'str', 'mom.um')
# Processing the call keyword arguments (line 1186)
kwargs_13623 = {}
# Getting the type of 'k' (line 1186)
k_13619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 22), 'k', False)
# Obtaining the member 'replace' of a type (line 1186)
replace_13620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 22), k_13619, 'replace')
# Calling replace(args, kwargs) (line 1186)
replace_call_result_13624 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 22), replace_13620, *[str_13621, str_13622], **kwargs_13623)

# Getting the type of '_aliases' (line 1186)
_aliases_13625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 8), '_aliases')
# Getting the type of 'k' (line 1186)
k_13626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 17), 'k')
# Storing an element on a container (line 1186)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1186, 8), _aliases_13625, (k_13626, replace_call_result_13624))
# SSA join for if statement (line 1185)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'ConstantWarning' class
# Getting the type of 'DeprecationWarning' (line 1189)
DeprecationWarning_13627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 22), 'DeprecationWarning')

class ConstantWarning(DeprecationWarning_13627, ):
    str_13628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1190, 4), 'str', 'Accessing a constant no longer in current CODATA data set')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1189, 0, False)
        # Assigning a type to the variable 'self' (line 1190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConstantWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConstantWarning' (line 1189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 0), 'ConstantWarning', ConstantWarning)

@norecursion
def _check_obsolete(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_obsolete'
    module_type_store = module_type_store.open_function_context('_check_obsolete', 1194, 0, False)
    
    # Passed parameters checking function
    _check_obsolete.stypy_localization = localization
    _check_obsolete.stypy_type_of_self = None
    _check_obsolete.stypy_type_store = module_type_store
    _check_obsolete.stypy_function_name = '_check_obsolete'
    _check_obsolete.stypy_param_names_list = ['key']
    _check_obsolete.stypy_varargs_param_name = None
    _check_obsolete.stypy_kwargs_param_name = None
    _check_obsolete.stypy_call_defaults = defaults
    _check_obsolete.stypy_call_varargs = varargs
    _check_obsolete.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_obsolete', ['key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_obsolete', localization, ['key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_obsolete(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'key' (line 1195)
    key_13629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 7), 'key')
    # Getting the type of '_obsolete_constants' (line 1195)
    _obsolete_constants_13630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 14), '_obsolete_constants')
    # Applying the binary operator 'in' (line 1195)
    result_contains_13631 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 7), 'in', key_13629, _obsolete_constants_13630)
    
    
    # Getting the type of 'key' (line 1195)
    key_13632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 38), 'key')
    # Getting the type of '_aliases' (line 1195)
    _aliases_13633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 49), '_aliases')
    # Applying the binary operator 'notin' (line 1195)
    result_contains_13634 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 38), 'notin', key_13632, _aliases_13633)
    
    # Applying the binary operator 'and' (line 1195)
    result_and_keyword_13635 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 7), 'and', result_contains_13631, result_contains_13634)
    
    # Testing the type of an if condition (line 1195)
    if_condition_13636 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1195, 4), result_and_keyword_13635)
    # Assigning a type to the variable 'if_condition_13636' (line 1195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 4), 'if_condition_13636', if_condition_13636)
    # SSA begins for if statement (line 1195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1196)
    # Processing the call arguments (line 1196)
    str_13639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1196, 22), 'str', "Constant '%s' is not in current %s data set")
    
    # Obtaining an instance of the builtin type 'tuple' (line 1197)
    tuple_13640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1197)
    # Adding element type (line 1197)
    # Getting the type of 'key' (line 1197)
    key_13641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 12), 'key', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 12), tuple_13640, key_13641)
    # Adding element type (line 1197)
    # Getting the type of '_current_codata' (line 1197)
    _current_codata_13642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 17), '_current_codata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 12), tuple_13640, _current_codata_13642)
    
    # Applying the binary operator '%' (line 1196)
    result_mod_13643 = python_operator(stypy.reporting.localization.Localization(__file__, 1196, 22), '%', str_13639, tuple_13640)
    
    # Getting the type of 'ConstantWarning' (line 1197)
    ConstantWarning_13644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 35), 'ConstantWarning', False)
    # Processing the call keyword arguments (line 1196)
    kwargs_13645 = {}
    # Getting the type of 'warnings' (line 1196)
    warnings_13637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1196)
    warn_13638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 8), warnings_13637, 'warn')
    # Calling warn(args, kwargs) (line 1196)
    warn_call_result_13646 = invoke(stypy.reporting.localization.Localization(__file__, 1196, 8), warn_13638, *[result_mod_13643, ConstantWarning_13644], **kwargs_13645)
    
    # SSA join for if statement (line 1195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_obsolete(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_obsolete' in the type store
    # Getting the type of 'stypy_return_type' (line 1194)
    stypy_return_type_13647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_obsolete'
    return stypy_return_type_13647

# Assigning a type to the variable '_check_obsolete' (line 1194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 0), '_check_obsolete', _check_obsolete)

@norecursion
def value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'value'
    module_type_store = module_type_store.open_function_context('value', 1200, 0, False)
    
    # Passed parameters checking function
    value.stypy_localization = localization
    value.stypy_type_of_self = None
    value.stypy_type_store = module_type_store
    value.stypy_function_name = 'value'
    value.stypy_param_names_list = ['key']
    value.stypy_varargs_param_name = None
    value.stypy_kwargs_param_name = None
    value.stypy_call_defaults = defaults
    value.stypy_call_varargs = varargs
    value.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'value', ['key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'value', localization, ['key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'value(...)' code ##################

    str_13648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, (-1)), 'str', "\n    Value in physical_constants indexed by key\n\n    Parameters\n    ----------\n    key : Python string or unicode\n        Key in dictionary `physical_constants`\n\n    Returns\n    -------\n    value : float\n        Value in `physical_constants` corresponding to `key`\n\n    See Also\n    --------\n    codata : Contains the description of `physical_constants`, which, as a\n        dictionary literal object, does not itself possess a docstring.\n\n    Examples\n    --------\n    >>> from scipy import constants\n    >>> constants.value(u'elementary charge')\n        1.6021766208e-19\n\n    ")
    
    # Call to _check_obsolete(...): (line 1226)
    # Processing the call arguments (line 1226)
    # Getting the type of 'key' (line 1226)
    key_13650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 20), 'key', False)
    # Processing the call keyword arguments (line 1226)
    kwargs_13651 = {}
    # Getting the type of '_check_obsolete' (line 1226)
    _check_obsolete_13649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 4), '_check_obsolete', False)
    # Calling _check_obsolete(args, kwargs) (line 1226)
    _check_obsolete_call_result_13652 = invoke(stypy.reporting.localization.Localization(__file__, 1226, 4), _check_obsolete_13649, *[key_13650], **kwargs_13651)
    
    
    # Obtaining the type of the subscript
    int_13653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 35), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 1227)
    key_13654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 30), 'key')
    # Getting the type of 'physical_constants' (line 1227)
    physical_constants_13655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 11), 'physical_constants')
    # Obtaining the member '__getitem__' of a type (line 1227)
    getitem___13656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 11), physical_constants_13655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1227)
    subscript_call_result_13657 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 11), getitem___13656, key_13654)
    
    # Obtaining the member '__getitem__' of a type (line 1227)
    getitem___13658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 11), subscript_call_result_13657, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1227)
    subscript_call_result_13659 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 11), getitem___13658, int_13653)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'stypy_return_type', subscript_call_result_13659)
    
    # ################# End of 'value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'value' in the type store
    # Getting the type of 'stypy_return_type' (line 1200)
    stypy_return_type_13660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13660)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'value'
    return stypy_return_type_13660

# Assigning a type to the variable 'value' (line 1200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1200, 0), 'value', value)

@norecursion
def unit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unit'
    module_type_store = module_type_store.open_function_context('unit', 1230, 0, False)
    
    # Passed parameters checking function
    unit.stypy_localization = localization
    unit.stypy_type_of_self = None
    unit.stypy_type_store = module_type_store
    unit.stypy_function_name = 'unit'
    unit.stypy_param_names_list = ['key']
    unit.stypy_varargs_param_name = None
    unit.stypy_kwargs_param_name = None
    unit.stypy_call_defaults = defaults
    unit.stypy_call_varargs = varargs
    unit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unit', ['key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unit', localization, ['key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unit(...)' code ##################

    str_13661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1255, (-1)), 'str', "\n    Unit in physical_constants indexed by key\n\n    Parameters\n    ----------\n    key : Python string or unicode\n        Key in dictionary `physical_constants`\n\n    Returns\n    -------\n    unit : Python string\n        Unit in `physical_constants` corresponding to `key`\n\n    See Also\n    --------\n    codata : Contains the description of `physical_constants`, which, as a\n        dictionary literal object, does not itself possess a docstring.\n\n    Examples\n    --------\n    >>> from scipy import constants\n    >>> constants.unit(u'proton mass')\n    'kg'\n\n    ")
    
    # Call to _check_obsolete(...): (line 1256)
    # Processing the call arguments (line 1256)
    # Getting the type of 'key' (line 1256)
    key_13663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 20), 'key', False)
    # Processing the call keyword arguments (line 1256)
    kwargs_13664 = {}
    # Getting the type of '_check_obsolete' (line 1256)
    _check_obsolete_13662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 4), '_check_obsolete', False)
    # Calling _check_obsolete(args, kwargs) (line 1256)
    _check_obsolete_call_result_13665 = invoke(stypy.reporting.localization.Localization(__file__, 1256, 4), _check_obsolete_13662, *[key_13663], **kwargs_13664)
    
    
    # Obtaining the type of the subscript
    int_13666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 35), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 1257)
    key_13667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 30), 'key')
    # Getting the type of 'physical_constants' (line 1257)
    physical_constants_13668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 11), 'physical_constants')
    # Obtaining the member '__getitem__' of a type (line 1257)
    getitem___13669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 11), physical_constants_13668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1257)
    subscript_call_result_13670 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 11), getitem___13669, key_13667)
    
    # Obtaining the member '__getitem__' of a type (line 1257)
    getitem___13671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 11), subscript_call_result_13670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1257)
    subscript_call_result_13672 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 11), getitem___13671, int_13666)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1257, 4), 'stypy_return_type', subscript_call_result_13672)
    
    # ################# End of 'unit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unit' in the type store
    # Getting the type of 'stypy_return_type' (line 1230)
    stypy_return_type_13673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13673)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unit'
    return stypy_return_type_13673

# Assigning a type to the variable 'unit' (line 1230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 0), 'unit', unit)

@norecursion
def precision(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'precision'
    module_type_store = module_type_store.open_function_context('precision', 1260, 0, False)
    
    # Passed parameters checking function
    precision.stypy_localization = localization
    precision.stypy_type_of_self = None
    precision.stypy_type_store = module_type_store
    precision.stypy_function_name = 'precision'
    precision.stypy_param_names_list = ['key']
    precision.stypy_varargs_param_name = None
    precision.stypy_kwargs_param_name = None
    precision.stypy_call_defaults = defaults
    precision.stypy_call_varargs = varargs
    precision.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'precision', ['key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'precision', localization, ['key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'precision(...)' code ##################

    str_13674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, (-1)), 'str', "\n    Relative precision in physical_constants indexed by key\n\n    Parameters\n    ----------\n    key : Python string or unicode\n        Key in dictionary `physical_constants`\n\n    Returns\n    -------\n    prec : float\n        Relative precision in `physical_constants` corresponding to `key`\n\n    See Also\n    --------\n    codata : Contains the description of `physical_constants`, which, as a\n        dictionary literal object, does not itself possess a docstring.\n\n    Examples\n    --------\n    >>> from scipy import constants\n    >>> constants.precision(u'proton mass')\n    1.2555138746605121e-08\n\n    ")
    
    # Call to _check_obsolete(...): (line 1286)
    # Processing the call arguments (line 1286)
    # Getting the type of 'key' (line 1286)
    key_13676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 20), 'key', False)
    # Processing the call keyword arguments (line 1286)
    kwargs_13677 = {}
    # Getting the type of '_check_obsolete' (line 1286)
    _check_obsolete_13675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 4), '_check_obsolete', False)
    # Calling _check_obsolete(args, kwargs) (line 1286)
    _check_obsolete_call_result_13678 = invoke(stypy.reporting.localization.Localization(__file__, 1286, 4), _check_obsolete_13675, *[key_13676], **kwargs_13677)
    
    
    # Obtaining the type of the subscript
    int_13679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 35), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 1287)
    key_13680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 30), 'key')
    # Getting the type of 'physical_constants' (line 1287)
    physical_constants_13681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 11), 'physical_constants')
    # Obtaining the member '__getitem__' of a type (line 1287)
    getitem___13682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 11), physical_constants_13681, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1287)
    subscript_call_result_13683 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 11), getitem___13682, key_13680)
    
    # Obtaining the member '__getitem__' of a type (line 1287)
    getitem___13684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 11), subscript_call_result_13683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1287)
    subscript_call_result_13685 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 11), getitem___13684, int_13679)
    
    
    # Obtaining the type of the subscript
    int_13686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 64), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 1287)
    key_13687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 59), 'key')
    # Getting the type of 'physical_constants' (line 1287)
    physical_constants_13688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 40), 'physical_constants')
    # Obtaining the member '__getitem__' of a type (line 1287)
    getitem___13689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 40), physical_constants_13688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1287)
    subscript_call_result_13690 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 40), getitem___13689, key_13687)
    
    # Obtaining the member '__getitem__' of a type (line 1287)
    getitem___13691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 40), subscript_call_result_13690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1287)
    subscript_call_result_13692 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 40), getitem___13691, int_13686)
    
    # Applying the binary operator 'div' (line 1287)
    result_div_13693 = python_operator(stypy.reporting.localization.Localization(__file__, 1287, 11), 'div', subscript_call_result_13685, subscript_call_result_13692)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 4), 'stypy_return_type', result_div_13693)
    
    # ################# End of 'precision(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'precision' in the type store
    # Getting the type of 'stypy_return_type' (line 1260)
    stypy_return_type_13694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13694)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'precision'
    return stypy_return_type_13694

# Assigning a type to the variable 'precision' (line 1260)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1260, 0), 'precision', precision)

@norecursion
def find(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1290)
    None_13695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 13), 'None')
    # Getting the type of 'False' (line 1290)
    False_13696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 24), 'False')
    defaults = [None_13695, False_13696]
    # Create a new context for function 'find'
    module_type_store = module_type_store.open_function_context('find', 1290, 0, False)
    
    # Passed parameters checking function
    find.stypy_localization = localization
    find.stypy_type_of_self = None
    find.stypy_type_store = module_type_store
    find.stypy_function_name = 'find'
    find.stypy_param_names_list = ['sub', 'disp']
    find.stypy_varargs_param_name = None
    find.stypy_kwargs_param_name = None
    find.stypy_call_defaults = defaults
    find.stypy_call_varargs = varargs
    find.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find', ['sub', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find', localization, ['sub', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find(...)' code ##################

    str_13697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1341, (-1)), 'str', "\n    Return list of physical_constant keys containing a given string.\n\n    Parameters\n    ----------\n    sub : str, unicode\n        Sub-string to search keys for.  By default, return all keys.\n    disp : bool\n        If True, print the keys that are found, and return None.\n        Otherwise, return the list of keys without printing anything.\n\n    Returns\n    -------\n    keys : list or None\n        If `disp` is False, the list of keys is returned.\n        Otherwise, None is returned.\n\n    See Also\n    --------\n    codata : Contains the description of `physical_constants`, which, as a\n        dictionary literal object, does not itself possess a docstring.\n\n    Examples\n    --------\n    >>> from scipy.constants import find, physical_constants\n\n    Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?\n\n    >>> find('boltzmann')\n    ['Boltzmann constant',\n     'Boltzmann constant in Hz/K',\n     'Boltzmann constant in eV/K',\n     'Boltzmann constant in inverse meters per kelvin',\n     'Stefan-Boltzmann constant']\n\n    Get the constant called 'Boltzmann constant in Hz/K':\n\n    >>> physical_constants['Boltzmann constant in Hz/K']\n    (20836612000.0, 'Hz K^-1', 12000.0)\n\n    Find constants with 'radius' in the key:\n\n    >>> find('radius')\n    ['Bohr radius',\n     'classical electron radius',\n     'deuteron rms charge radius',\n     'proton rms charge radius']\n    >>> physical_constants['classical electron radius']\n    (2.8179403227e-15, 'm', 1.9e-24)\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 1342)
    # Getting the type of 'sub' (line 1342)
    sub_13698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1342, 7), 'sub')
    # Getting the type of 'None' (line 1342)
    None_13699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1342, 14), 'None')
    
    (may_be_13700, more_types_in_union_13701) = may_be_none(sub_13698, None_13699)

    if may_be_13700:

        if more_types_in_union_13701:
            # Runtime conditional SSA (line 1342)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1343):
        
        # Call to list(...): (line 1343)
        # Processing the call arguments (line 1343)
        
        # Call to keys(...): (line 1343)
        # Processing the call keyword arguments (line 1343)
        kwargs_13705 = {}
        # Getting the type of '_current_constants' (line 1343)
        _current_constants_13703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 22), '_current_constants', False)
        # Obtaining the member 'keys' of a type (line 1343)
        keys_13704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1343, 22), _current_constants_13703, 'keys')
        # Calling keys(args, kwargs) (line 1343)
        keys_call_result_13706 = invoke(stypy.reporting.localization.Localization(__file__, 1343, 22), keys_13704, *[], **kwargs_13705)
        
        # Processing the call keyword arguments (line 1343)
        kwargs_13707 = {}
        # Getting the type of 'list' (line 1343)
        list_13702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 17), 'list', False)
        # Calling list(args, kwargs) (line 1343)
        list_call_result_13708 = invoke(stypy.reporting.localization.Localization(__file__, 1343, 17), list_13702, *[keys_call_result_13706], **kwargs_13707)
        
        # Assigning a type to the variable 'result' (line 1343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1343, 8), 'result', list_call_result_13708)

        if more_types_in_union_13701:
            # Runtime conditional SSA for else branch (line 1342)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13700) or more_types_in_union_13701):
        
        # Assigning a ListComp to a Name (line 1345):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of '_current_constants' (line 1345)
        _current_constants_13719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 33), '_current_constants')
        comprehension_13720 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 18), _current_constants_13719)
        # Assigning a type to the variable 'key' (line 1345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 18), 'key', comprehension_13720)
        
        
        # Call to lower(...): (line 1346)
        # Processing the call keyword arguments (line 1346)
        kwargs_13712 = {}
        # Getting the type of 'sub' (line 1346)
        sub_13710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1346, 21), 'sub', False)
        # Obtaining the member 'lower' of a type (line 1346)
        lower_13711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1346, 21), sub_13710, 'lower')
        # Calling lower(args, kwargs) (line 1346)
        lower_call_result_13713 = invoke(stypy.reporting.localization.Localization(__file__, 1346, 21), lower_13711, *[], **kwargs_13712)
        
        
        # Call to lower(...): (line 1346)
        # Processing the call keyword arguments (line 1346)
        kwargs_13716 = {}
        # Getting the type of 'key' (line 1346)
        key_13714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1346, 36), 'key', False)
        # Obtaining the member 'lower' of a type (line 1346)
        lower_13715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1346, 36), key_13714, 'lower')
        # Calling lower(args, kwargs) (line 1346)
        lower_call_result_13717 = invoke(stypy.reporting.localization.Localization(__file__, 1346, 36), lower_13715, *[], **kwargs_13716)
        
        # Applying the binary operator 'in' (line 1346)
        result_contains_13718 = python_operator(stypy.reporting.localization.Localization(__file__, 1346, 21), 'in', lower_call_result_13713, lower_call_result_13717)
        
        # Getting the type of 'key' (line 1345)
        key_13709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 18), 'key')
        list_13721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 18), list_13721, key_13709)
        # Assigning a type to the variable 'result' (line 1345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1345, 8), 'result', list_13721)

        if (may_be_13700 and more_types_in_union_13701):
            # SSA join for if statement (line 1342)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to sort(...): (line 1348)
    # Processing the call keyword arguments (line 1348)
    kwargs_13724 = {}
    # Getting the type of 'result' (line 1348)
    result_13722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 4), 'result', False)
    # Obtaining the member 'sort' of a type (line 1348)
    sort_13723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1348, 4), result_13722, 'sort')
    # Calling sort(args, kwargs) (line 1348)
    sort_call_result_13725 = invoke(stypy.reporting.localization.Localization(__file__, 1348, 4), sort_13723, *[], **kwargs_13724)
    
    
    # Getting the type of 'disp' (line 1349)
    disp_13726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 7), 'disp')
    # Testing the type of an if condition (line 1349)
    if_condition_13727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1349, 4), disp_13726)
    # Assigning a type to the variable 'if_condition_13727' (line 1349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1349, 4), 'if_condition_13727', if_condition_13727)
    # SSA begins for if statement (line 1349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'result' (line 1350)
    result_13728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 19), 'result')
    # Testing the type of a for loop iterable (line 1350)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1350, 8), result_13728)
    # Getting the type of the for loop variable (line 1350)
    for_loop_var_13729 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1350, 8), result_13728)
    # Assigning a type to the variable 'key' (line 1350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1350, 8), 'key', for_loop_var_13729)
    # SSA begins for a for statement (line 1350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 1351)
    # Processing the call arguments (line 1351)
    # Getting the type of 'key' (line 1351)
    key_13731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 18), 'key', False)
    # Processing the call keyword arguments (line 1351)
    kwargs_13732 = {}
    # Getting the type of 'print' (line 1351)
    print_13730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 12), 'print', False)
    # Calling print(args, kwargs) (line 1351)
    print_call_result_13733 = invoke(stypy.reporting.localization.Localization(__file__, 1351, 12), print_13730, *[key_13731], **kwargs_13732)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 1352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 8), 'stypy_return_type', types.NoneType)
    # SSA branch for the else part of an if statement (line 1349)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'result' (line 1354)
    result_13734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 1354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1354, 8), 'stypy_return_type', result_13734)
    # SSA join for if statement (line 1349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'find(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find' in the type store
    # Getting the type of 'stypy_return_type' (line 1290)
    stypy_return_type_13735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13735)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find'
    return stypy_return_type_13735

# Assigning a type to the variable 'find' (line 1290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 0), 'find', find)

# Assigning a Call to a Name (line 1357):

# Call to value(...): (line 1357)
# Processing the call arguments (line 1357)
str_13737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1357, 10), 'str', 'speed of light in vacuum')
# Processing the call keyword arguments (line 1357)
kwargs_13738 = {}
# Getting the type of 'value' (line 1357)
value_13736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 4), 'value', False)
# Calling value(args, kwargs) (line 1357)
value_call_result_13739 = invoke(stypy.reporting.localization.Localization(__file__, 1357, 4), value_13736, *[str_13737], **kwargs_13738)

# Assigning a type to the variable 'c' (line 1357)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1357, 0), 'c', value_call_result_13739)

# Assigning a BinOp to a Name (line 1358):
float_13740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 6), 'float')
# Getting the type of 'pi' (line 1358)
pi_13741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 13), 'pi')
# Applying the binary operator '*' (line 1358)
result_mul_13742 = python_operator(stypy.reporting.localization.Localization(__file__, 1358, 6), '*', float_13740, pi_13741)

# Assigning a type to the variable 'mu0' (line 1358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 0), 'mu0', result_mul_13742)

# Assigning a BinOp to a Name (line 1359):
int_13743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 11), 'int')
# Getting the type of 'mu0' (line 1359)
mu0_13744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 16), 'mu0')
# Getting the type of 'c' (line 1359)
c_13745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 22), 'c')
# Applying the binary operator '*' (line 1359)
result_mul_13746 = python_operator(stypy.reporting.localization.Localization(__file__, 1359, 16), '*', mu0_13744, c_13745)

# Getting the type of 'c' (line 1359)
c_13747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 26), 'c')
# Applying the binary operator '*' (line 1359)
result_mul_13748 = python_operator(stypy.reporting.localization.Localization(__file__, 1359, 24), '*', result_mul_13746, c_13747)

# Applying the binary operator 'div' (line 1359)
result_div_13749 = python_operator(stypy.reporting.localization.Localization(__file__, 1359, 11), 'div', int_13743, result_mul_13748)

# Assigning a type to the variable 'epsilon0' (line 1359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 0), 'epsilon0', result_div_13749)

# Assigning a Dict to a Name (line 1361):

# Obtaining an instance of the builtin type 'dict' (line 1361)
dict_13750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1361, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1361)
# Adding element type (key, value) (line 1361)
str_13751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 4), 'str', 'mag. constant')

# Obtaining an instance of the builtin type 'tuple' (line 1362)
tuple_13752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1362)
# Adding element type (line 1362)
# Getting the type of 'mu0' (line 1362)
mu0_13753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 22), 'mu0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1362, 22), tuple_13752, mu0_13753)
# Adding element type (line 1362)
str_13754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 27), 'str', 'N A^-2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1362, 22), tuple_13752, str_13754)
# Adding element type (line 1362)
float_13755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1362, 22), tuple_13752, float_13755)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13751, tuple_13752))
# Adding element type (key, value) (line 1361)
str_13756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1363, 4), 'str', 'electric constant')

# Obtaining an instance of the builtin type 'tuple' (line 1363)
tuple_13757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1363, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1363)
# Adding element type (line 1363)
# Getting the type of 'epsilon0' (line 1363)
epsilon0_13758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 26), 'epsilon0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1363, 26), tuple_13757, epsilon0_13758)
# Adding element type (line 1363)
str_13759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1363, 36), 'str', 'F m^-1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1363, 26), tuple_13757, str_13759)
# Adding element type (line 1363)
float_13760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1363, 46), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1363, 26), tuple_13757, float_13760)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13756, tuple_13757))
# Adding element type (key, value) (line 1361)
str_13761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 4), 'str', 'characteristic impedance of vacuum')

# Obtaining an instance of the builtin type 'tuple' (line 1364)
tuple_13762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1364)
# Adding element type (line 1364)

# Call to sqrt(...): (line 1364)
# Processing the call arguments (line 1364)
# Getting the type of 'mu0' (line 1364)
mu0_13764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 48), 'mu0', False)
# Getting the type of 'epsilon0' (line 1364)
epsilon0_13765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 54), 'epsilon0', False)
# Applying the binary operator 'div' (line 1364)
result_div_13766 = python_operator(stypy.reporting.localization.Localization(__file__, 1364, 48), 'div', mu0_13764, epsilon0_13765)

# Processing the call keyword arguments (line 1364)
kwargs_13767 = {}
# Getting the type of 'sqrt' (line 1364)
sqrt_13763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 43), 'sqrt', False)
# Calling sqrt(args, kwargs) (line 1364)
sqrt_call_result_13768 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 43), sqrt_13763, *[result_div_13766], **kwargs_13767)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 43), tuple_13762, sqrt_call_result_13768)
# Adding element type (line 1364)
str_13769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 65), 'str', 'ohm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 43), tuple_13762, str_13769)
# Adding element type (line 1364)
float_13770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 72), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 43), tuple_13762, float_13770)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13761, tuple_13762))
# Adding element type (key, value) (line 1361)
str_13771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 4), 'str', 'atomic unit of permittivity')

# Obtaining an instance of the builtin type 'tuple' (line 1365)
tuple_13772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1365)
# Adding element type (line 1365)
int_13773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 36), 'int')
# Getting the type of 'epsilon0' (line 1365)
epsilon0_13774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 40), 'epsilon0')
# Applying the binary operator '*' (line 1365)
result_mul_13775 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 36), '*', int_13773, epsilon0_13774)

# Getting the type of 'pi' (line 1365)
pi_13776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 51), 'pi')
# Applying the binary operator '*' (line 1365)
result_mul_13777 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 49), '*', result_mul_13775, pi_13776)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1365, 36), tuple_13772, result_mul_13777)
# Adding element type (line 1365)
str_13778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 55), 'str', 'F m^-1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1365, 36), tuple_13772, str_13778)
# Adding element type (line 1365)
float_13779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 65), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1365, 36), tuple_13772, float_13779)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13771, tuple_13772))
# Adding element type (key, value) (line 1361)
str_13780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 4), 'str', 'joule-kilogram relationship')

# Obtaining an instance of the builtin type 'tuple' (line 1366)
tuple_13781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1366)
# Adding element type (line 1366)
int_13782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 36), 'int')
# Getting the type of 'c' (line 1366)
c_13783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 41), 'c')
# Getting the type of 'c' (line 1366)
c_13784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 45), 'c')
# Applying the binary operator '*' (line 1366)
result_mul_13785 = python_operator(stypy.reporting.localization.Localization(__file__, 1366, 41), '*', c_13783, c_13784)

# Applying the binary operator 'div' (line 1366)
result_div_13786 = python_operator(stypy.reporting.localization.Localization(__file__, 1366, 36), 'div', int_13782, result_mul_13785)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1366, 36), tuple_13781, result_div_13786)
# Adding element type (line 1366)
str_13787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 49), 'str', 'kg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1366, 36), tuple_13781, str_13787)
# Adding element type (line 1366)
float_13788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1366, 36), tuple_13781, float_13788)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13780, tuple_13781))
# Adding element type (key, value) (line 1361)
str_13789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 4), 'str', 'kilogram-joule relationship')

# Obtaining an instance of the builtin type 'tuple' (line 1367)
tuple_13790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1367)
# Adding element type (line 1367)
# Getting the type of 'c' (line 1367)
c_13791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 36), 'c')
# Getting the type of 'c' (line 1367)
c_13792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 40), 'c')
# Applying the binary operator '*' (line 1367)
result_mul_13793 = python_operator(stypy.reporting.localization.Localization(__file__, 1367, 36), '*', c_13791, c_13792)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1367, 36), tuple_13790, result_mul_13793)
# Adding element type (line 1367)
str_13794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 43), 'str', 'J')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1367, 36), tuple_13790, str_13794)
# Adding element type (line 1367)
float_13795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1367, 36), tuple_13790, float_13795)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13789, tuple_13790))
# Adding element type (key, value) (line 1361)
str_13796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'str', 'hertz-inverse meter relationship')

# Obtaining an instance of the builtin type 'tuple' (line 1368)
tuple_13797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1368)
# Adding element type (line 1368)
int_13798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 41), 'int')
# Getting the type of 'c' (line 1368)
c_13799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 45), 'c')
# Applying the binary operator 'div' (line 1368)
result_div_13800 = python_operator(stypy.reporting.localization.Localization(__file__, 1368, 41), 'div', int_13798, c_13799)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1368, 41), tuple_13797, result_div_13800)
# Adding element type (line 1368)
str_13801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 48), 'str', 'm^-1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1368, 41), tuple_13797, str_13801)
# Adding element type (line 1368)
float_13802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1368, 41), tuple_13797, float_13802)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 15), dict_13750, (str_13796, tuple_13797))

# Assigning a type to the variable 'exact_values' (line 1361)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 0), 'exact_values', dict_13750)

# Getting the type of 'exact_values' (line 1372)
exact_values_13803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 11), 'exact_values')
# Testing the type of a for loop iterable (line 1372)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1372, 0), exact_values_13803)
# Getting the type of the for loop variable (line 1372)
for_loop_var_13804 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1372, 0), exact_values_13803)
# Assigning a type to the variable 'key' (line 1372)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1372, 0), 'key', for_loop_var_13804)
# SSA begins for a for statement (line 1372)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Name (line 1373):

# Obtaining the type of the subscript
int_13805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1373, 34), 'int')

# Obtaining the type of the subscript
# Getting the type of 'key' (line 1373)
key_13806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 29), 'key')
# Getting the type of '_current_constants' (line 1373)
_current_constants_13807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 10), '_current_constants')
# Obtaining the member '__getitem__' of a type (line 1373)
getitem___13808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1373, 10), _current_constants_13807, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1373)
subscript_call_result_13809 = invoke(stypy.reporting.localization.Localization(__file__, 1373, 10), getitem___13808, key_13806)

# Obtaining the member '__getitem__' of a type (line 1373)
getitem___13810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1373, 10), subscript_call_result_13809, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1373)
subscript_call_result_13811 = invoke(stypy.reporting.localization.Localization(__file__, 1373, 10), getitem___13810, int_13805)

# Assigning a type to the variable 'val' (line 1373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1373, 4), 'val', subscript_call_result_13811)



# Call to abs(...): (line 1374)
# Processing the call arguments (line 1374)

# Obtaining the type of the subscript
int_13813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1374, 29), 'int')

# Obtaining the type of the subscript
# Getting the type of 'key' (line 1374)
key_13814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 24), 'key', False)
# Getting the type of 'exact_values' (line 1374)
exact_values_13815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 11), 'exact_values', False)
# Obtaining the member '__getitem__' of a type (line 1374)
getitem___13816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1374, 11), exact_values_13815, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1374)
subscript_call_result_13817 = invoke(stypy.reporting.localization.Localization(__file__, 1374, 11), getitem___13816, key_13814)

# Obtaining the member '__getitem__' of a type (line 1374)
getitem___13818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1374, 11), subscript_call_result_13817, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1374)
subscript_call_result_13819 = invoke(stypy.reporting.localization.Localization(__file__, 1374, 11), getitem___13818, int_13813)

# Getting the type of 'val' (line 1374)
val_13820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 34), 'val', False)
# Applying the binary operator '-' (line 1374)
result_sub_13821 = python_operator(stypy.reporting.localization.Localization(__file__, 1374, 11), '-', subscript_call_result_13819, val_13820)

# Processing the call keyword arguments (line 1374)
kwargs_13822 = {}
# Getting the type of 'abs' (line 1374)
abs_13812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 7), 'abs', False)
# Calling abs(args, kwargs) (line 1374)
abs_call_result_13823 = invoke(stypy.reporting.localization.Localization(__file__, 1374, 7), abs_13812, *[result_sub_13821], **kwargs_13822)

# Getting the type of 'val' (line 1374)
val_13824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 41), 'val')
# Applying the binary operator 'div' (line 1374)
result_div_13825 = python_operator(stypy.reporting.localization.Localization(__file__, 1374, 7), 'div', abs_call_result_13823, val_13824)

float_13826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1374, 47), 'float')
# Applying the binary operator '>' (line 1374)
result_gt_13827 = python_operator(stypy.reporting.localization.Localization(__file__, 1374, 7), '>', result_div_13825, float_13826)

# Testing the type of an if condition (line 1374)
if_condition_13828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1374, 4), result_gt_13827)
# Assigning a type to the variable 'if_condition_13828' (line 1374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1374, 4), 'if_condition_13828', if_condition_13828)
# SSA begins for if statement (line 1374)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to ValueError(...): (line 1375)
# Processing the call arguments (line 1375)
str_13830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1375, 25), 'str', 'Constants.codata: exact values too far off.')
# Processing the call keyword arguments (line 1375)
kwargs_13831 = {}
# Getting the type of 'ValueError' (line 1375)
ValueError_13829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1375, 14), 'ValueError', False)
# Calling ValueError(args, kwargs) (line 1375)
ValueError_call_result_13832 = invoke(stypy.reporting.localization.Localization(__file__, 1375, 14), ValueError_13829, *[str_13830], **kwargs_13831)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1375, 8), ValueError_call_result_13832, 'raise parameter', BaseException)
# SSA join for if statement (line 1374)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Call to update(...): (line 1377)
# Processing the call arguments (line 1377)
# Getting the type of 'exact_values' (line 1377)
exact_values_13835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 26), 'exact_values', False)
# Processing the call keyword arguments (line 1377)
kwargs_13836 = {}
# Getting the type of 'physical_constants' (line 1377)
physical_constants_13833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 0), 'physical_constants', False)
# Obtaining the member 'update' of a type (line 1377)
update_13834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1377, 0), physical_constants_13833, 'update')
# Calling update(args, kwargs) (line 1377)
update_call_result_13837 = invoke(stypy.reporting.localization.Localization(__file__, 1377, 0), update_13834, *[exact_values_13835], **kwargs_13836)



# Call to list(...): (line 1380)
# Processing the call arguments (line 1380)

# Call to items(...): (line 1380)
# Processing the call keyword arguments (line 1380)
kwargs_13841 = {}
# Getting the type of '_aliases' (line 1380)
_aliases_13839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 17), '_aliases', False)
# Obtaining the member 'items' of a type (line 1380)
items_13840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 17), _aliases_13839, 'items')
# Calling items(args, kwargs) (line 1380)
items_call_result_13842 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 17), items_13840, *[], **kwargs_13841)

# Processing the call keyword arguments (line 1380)
kwargs_13843 = {}
# Getting the type of 'list' (line 1380)
list_13838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 12), 'list', False)
# Calling list(args, kwargs) (line 1380)
list_call_result_13844 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 12), list_13838, *[items_call_result_13842], **kwargs_13843)

# Testing the type of a for loop iterable (line 1380)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1380, 0), list_call_result_13844)
# Getting the type of the for loop variable (line 1380)
for_loop_var_13845 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1380, 0), list_call_result_13844)
# Assigning a type to the variable 'k' (line 1380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 0), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1380, 0), for_loop_var_13845))
# Assigning a type to the variable 'v' (line 1380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 0), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1380, 0), for_loop_var_13845))
# SSA begins for a for statement (line 1380)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Getting the type of 'v' (line 1381)
v_13846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 7), 'v')
# Getting the type of '_current_constants' (line 1381)
_current_constants_13847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 12), '_current_constants')
# Applying the binary operator 'in' (line 1381)
result_contains_13848 = python_operator(stypy.reporting.localization.Localization(__file__, 1381, 7), 'in', v_13846, _current_constants_13847)

# Testing the type of an if condition (line 1381)
if_condition_13849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1381, 4), result_contains_13848)
# Assigning a type to the variable 'if_condition_13849' (line 1381)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 4), 'if_condition_13849', if_condition_13849)
# SSA begins for if statement (line 1381)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Subscript to a Subscript (line 1382):

# Obtaining the type of the subscript
# Getting the type of 'v' (line 1382)
v_13850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 51), 'v')
# Getting the type of 'physical_constants' (line 1382)
physical_constants_13851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 32), 'physical_constants')
# Obtaining the member '__getitem__' of a type (line 1382)
getitem___13852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1382, 32), physical_constants_13851, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1382)
subscript_call_result_13853 = invoke(stypy.reporting.localization.Localization(__file__, 1382, 32), getitem___13852, v_13850)

# Getting the type of 'physical_constants' (line 1382)
physical_constants_13854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 8), 'physical_constants')
# Getting the type of 'k' (line 1382)
k_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 27), 'k')
# Storing an element on a container (line 1382)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1382, 8), physical_constants_13854, (k_13855, subscript_call_result_13853))
# SSA branch for the else part of an if statement (line 1381)
module_type_store.open_ssa_branch('else')
# Deleting a member
# Getting the type of '_aliases' (line 1384)
_aliases_13856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 12), '_aliases')

# Obtaining the type of the subscript
# Getting the type of 'k' (line 1384)
k_13857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 21), 'k')
# Getting the type of '_aliases' (line 1384)
_aliases_13858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 12), '_aliases')
# Obtaining the member '__getitem__' of a type (line 1384)
getitem___13859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1384, 12), _aliases_13858, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1384)
subscript_call_result_13860 = invoke(stypy.reporting.localization.Localization(__file__, 1384, 12), getitem___13859, k_13857)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1384, 8), _aliases_13856, subscript_call_result_13860)
# SSA join for if statement (line 1381)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
