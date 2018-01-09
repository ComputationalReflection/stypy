
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: 
3: =============================
4:  Byteswapping and byte order
5: =============================
6: 
7: Introduction to byte ordering and ndarrays
8: ==========================================
9: 
10: The ``ndarray`` is an object that provide a python array interface to data
11: in memory.
12: 
13: It often happens that the memory that you want to view with an array is
14: not of the same byte ordering as the computer on which you are running
15: Python.
16: 
17: For example, I might be working on a computer with a little-endian CPU -
18: such as an Intel Pentium, but I have loaded some data from a file
19: written by a computer that is big-endian.  Let's say I have loaded 4
20: bytes from a file written by a Sun (big-endian) computer.  I know that
21: these 4 bytes represent two 16-bit integers.  On a big-endian machine, a
22: two-byte integer is stored with the Most Significant Byte (MSB) first,
23: and then the Least Significant Byte (LSB). Thus the bytes are, in memory order:
24: 
25: #. MSB integer 1
26: #. LSB integer 1
27: #. MSB integer 2
28: #. LSB integer 2
29: 
30: Let's say the two integers were in fact 1 and 770.  Because 770 = 256 *
31: 3 + 2, the 4 bytes in memory would contain respectively: 0, 1, 3, 2.
32: The bytes I have loaded from the file would have these contents:
33: 
34: >>> big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
35: >>> big_end_str
36: '\\x00\\x01\\x03\\x02'
37: 
38: We might want to use an ``ndarray`` to access these integers.  In that
39: case, we can create an array around this memory, and tell numpy that
40: there are two integers, and that they are 16 bit and big-endian:
41: 
42: >>> import numpy as np
43: >>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_str)
44: >>> big_end_arr[0]
45: 1
46: >>> big_end_arr[1]
47: 770
48: 
49: Note the array ``dtype`` above of ``>i2``.  The ``>`` means 'big-endian'
50: (``<`` is little-endian) and ``i2`` means 'signed 2-byte integer'.  For
51: example, if our data represented a single unsigned 4-byte little-endian
52: integer, the dtype string would be ``<u4``.
53: 
54: In fact, why don't we try that?
55: 
56: >>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_str)
57: >>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3
58: True
59: 
60: Returning to our ``big_end_arr`` - in this case our underlying data is
61: big-endian (data endianness) and we've set the dtype to match (the dtype
62: is also big-endian).  However, sometimes you need to flip these around.
63: 
64: .. warning::
65: 
66:     Scalars currently do not include byte order information, so extracting
67:     a scalar from an array will return an integer in native byte order.
68:     Hence:
69: 
70:     >>> big_end_arr[0].dtype.byteorder == little_end_u4[0].dtype.byteorder
71:     True
72: 
73: Changing byte ordering
74: ======================
75: 
76: As you can imagine from the introduction, there are two ways you can
77: affect the relationship between the byte ordering of the array and the
78: underlying memory it is looking at:
79: 
80: * Change the byte-ordering information in the array dtype so that it
81:   interprets the undelying data as being in a different byte order.
82:   This is the role of ``arr.newbyteorder()``
83: * Change the byte-ordering of the underlying data, leaving the dtype
84:   interpretation as it was.  This is what ``arr.byteswap()`` does.
85: 
86: The common situations in which you need to change byte ordering are:
87: 
88: #. Your data and dtype endianess don't match, and you want to change
89:    the dtype so that it matches the data.
90: #. Your data and dtype endianess don't match, and you want to swap the
91:    data so that they match the dtype
92: #. Your data and dtype endianess match, but you want the data swapped
93:    and the dtype to reflect this
94: 
95: Data and dtype endianness don't match, change dtype to match data
96: -----------------------------------------------------------------
97: 
98: We make something where they don't match:
99: 
100: >>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_str)
101: >>> wrong_end_dtype_arr[0]
102: 256
103: 
104: The obvious fix for this situation is to change the dtype so it gives
105: the correct endianness:
106: 
107: >>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
108: >>> fixed_end_dtype_arr[0]
109: 1
110: 
111: Note the the array has not changed in memory:
112: 
113: >>> fixed_end_dtype_arr.tobytes() == big_end_str
114: True
115: 
116: Data and type endianness don't match, change data to match dtype
117: ----------------------------------------------------------------
118: 
119: You might want to do this if you need the data in memory to be a certain
120: ordering.  For example you might be writing the memory out to a file
121: that needs a certain byte ordering.
122: 
123: >>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
124: >>> fixed_end_mem_arr[0]
125: 1
126: 
127: Now the array *has* changed in memory:
128: 
129: >>> fixed_end_mem_arr.tobytes() == big_end_str
130: False
131: 
132: Data and dtype endianness match, swap data and dtype
133: ----------------------------------------------------
134: 
135: You may have a correctly specified array dtype, but you need the array
136: to have the opposite byte order in memory, and you want the dtype to
137: match so the array values make sense.  In this case you just do both of
138: the previous operations:
139: 
140: >>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()
141: >>> swapped_end_arr[0]
142: 1
143: >>> swapped_end_arr.tobytes() == big_end_str
144: False
145: 
146: An easier way of casting the data to a specific dtype and byte ordering
147: can be achieved with the ndarray astype method:
148: 
149: >>> swapped_end_arr = big_end_arr.astype('<i2')
150: >>> swapped_end_arr[0]
151: 1
152: >>> swapped_end_arr.tobytes() == big_end_str
153: False
154: 
155: '''
156: from __future__ import division, absolute_import, print_function
157: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', "\n\n=============================\n Byteswapping and byte order\n=============================\n\nIntroduction to byte ordering and ndarrays\n==========================================\n\nThe ``ndarray`` is an object that provide a python array interface to data\nin memory.\n\nIt often happens that the memory that you want to view with an array is\nnot of the same byte ordering as the computer on which you are running\nPython.\n\nFor example, I might be working on a computer with a little-endian CPU -\nsuch as an Intel Pentium, but I have loaded some data from a file\nwritten by a computer that is big-endian.  Let's say I have loaded 4\nbytes from a file written by a Sun (big-endian) computer.  I know that\nthese 4 bytes represent two 16-bit integers.  On a big-endian machine, a\ntwo-byte integer is stored with the Most Significant Byte (MSB) first,\nand then the Least Significant Byte (LSB). Thus the bytes are, in memory order:\n\n#. MSB integer 1\n#. LSB integer 1\n#. MSB integer 2\n#. LSB integer 2\n\nLet's say the two integers were in fact 1 and 770.  Because 770 = 256 *\n3 + 2, the 4 bytes in memory would contain respectively: 0, 1, 3, 2.\nThe bytes I have loaded from the file would have these contents:\n\n>>> big_end_str = chr(0) + chr(1) + chr(3) + chr(2)\n>>> big_end_str\n'\\x00\\x01\\x03\\x02'\n\nWe might want to use an ``ndarray`` to access these integers.  In that\ncase, we can create an array around this memory, and tell numpy that\nthere are two integers, and that they are 16 bit and big-endian:\n\n>>> import numpy as np\n>>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_str)\n>>> big_end_arr[0]\n1\n>>> big_end_arr[1]\n770\n\nNote the array ``dtype`` above of ``>i2``.  The ``>`` means 'big-endian'\n(``<`` is little-endian) and ``i2`` means 'signed 2-byte integer'.  For\nexample, if our data represented a single unsigned 4-byte little-endian\ninteger, the dtype string would be ``<u4``.\n\nIn fact, why don't we try that?\n\n>>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_str)\n>>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3\nTrue\n\nReturning to our ``big_end_arr`` - in this case our underlying data is\nbig-endian (data endianness) and we've set the dtype to match (the dtype\nis also big-endian).  However, sometimes you need to flip these around.\n\n.. warning::\n\n    Scalars currently do not include byte order information, so extracting\n    a scalar from an array will return an integer in native byte order.\n    Hence:\n\n    >>> big_end_arr[0].dtype.byteorder == little_end_u4[0].dtype.byteorder\n    True\n\nChanging byte ordering\n======================\n\nAs you can imagine from the introduction, there are two ways you can\naffect the relationship between the byte ordering of the array and the\nunderlying memory it is looking at:\n\n* Change the byte-ordering information in the array dtype so that it\n  interprets the undelying data as being in a different byte order.\n  This is the role of ``arr.newbyteorder()``\n* Change the byte-ordering of the underlying data, leaving the dtype\n  interpretation as it was.  This is what ``arr.byteswap()`` does.\n\nThe common situations in which you need to change byte ordering are:\n\n#. Your data and dtype endianess don't match, and you want to change\n   the dtype so that it matches the data.\n#. Your data and dtype endianess don't match, and you want to swap the\n   data so that they match the dtype\n#. Your data and dtype endianess match, but you want the data swapped\n   and the dtype to reflect this\n\nData and dtype endianness don't match, change dtype to match data\n-----------------------------------------------------------------\n\nWe make something where they don't match:\n\n>>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_str)\n>>> wrong_end_dtype_arr[0]\n256\n\nThe obvious fix for this situation is to change the dtype so it gives\nthe correct endianness:\n\n>>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()\n>>> fixed_end_dtype_arr[0]\n1\n\nNote the the array has not changed in memory:\n\n>>> fixed_end_dtype_arr.tobytes() == big_end_str\nTrue\n\nData and type endianness don't match, change data to match dtype\n----------------------------------------------------------------\n\nYou might want to do this if you need the data in memory to be a certain\nordering.  For example you might be writing the memory out to a file\nthat needs a certain byte ordering.\n\n>>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()\n>>> fixed_end_mem_arr[0]\n1\n\nNow the array *has* changed in memory:\n\n>>> fixed_end_mem_arr.tobytes() == big_end_str\nFalse\n\nData and dtype endianness match, swap data and dtype\n----------------------------------------------------\n\nYou may have a correctly specified array dtype, but you need the array\nto have the opposite byte order in memory, and you want the dtype to\nmatch so the array values make sense.  In this case you just do both of\nthe previous operations:\n\n>>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()\n>>> swapped_end_arr[0]\n1\n>>> swapped_end_arr.tobytes() == big_end_str\nFalse\n\nAn easier way of casting the data to a specific dtype and byte ordering\ncan be achieved with the ndarray astype method:\n\n>>> swapped_end_arr = big_end_arr.astype('<i2')\n>>> swapped_end_arr[0]\n1\n>>> swapped_end_arr.tobytes() == big_end_str\nFalse\n\n")

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
