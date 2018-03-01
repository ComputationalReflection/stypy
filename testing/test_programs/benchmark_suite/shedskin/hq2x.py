# hq2x filter demo program
# ----------------------------------------------------------
# Copyright (C) 2003 MaxSt ( maxst@hiend3d.com )
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

# [28/02/2018] Refactorization of code to improve coding style

import sys, os


def Relative(path):
    return os.path.join(os.path.dirname(__file__), path)


LUT16to32 = 65536 * [0]
RGBtoYUV = 65536 * [0]

Ymask = 0x00FF0000
Umask = 0x0000FF00
Vmask = 0x000000FF
trY = 0x00300000
trU = 0x00000700
trV = 0x00000006


class PPM:
    def __init__(self, w, h, rgb=None):
        self.w, self.h = w, h
        if rgb:
            self.rgb = rgb
        else:
            self.rgb = [0 for i in range(w * h)]

    @staticmethod
    def load(filename):
        lines = [l.strip() for l in file(filename)]

        assert lines[0] == 'P3'
        w, h = map(int, lines[1].split())
        assert int(lines[2]) == 255
        values = []
        for line in lines[3:]:
            values.extend(map(int, line.split()))
        rgb = []
        for i in range(0, len(values), 3):
            r = values[i] >> 3
            g = values[i + 1] >> 2
            b = values[i + 2] >> 3
            rgb.append(r << 11 | g << 5 | b)
        return PPM(w, h, rgb)

    def save(self, filename):
        f = file(filename, 'w')
        print >> f, 'P3'
        print >> f, self.w, self.h
        print >> f, '255'
        for rgb in self.rgb:
            r = ((rgb >> 16) & 0xff)
            g = ((rgb >> 8) & 0xff)
            b = (rgb & 0xff)
            print >> f, r, g, b
        print >> f
        f.close()


def diff(w1, w2):
    YUV1 = RGBtoYUV[w1]
    YUV2 = RGBtoYUV[w2]
    return (abs((YUV1 & Ymask) - (YUV2 & Ymask)) > trY) or \
           (abs((YUV1 & Umask) - (YUV2 & Umask)) > trU) or \
           (abs((YUV1 & Vmask) - (YUV2 & Vmask)) > trV)


def Interp1(c1, c2):
    return (c1 * 3 + c2) >> 2


def Interp2(c1, c2, c3):
    return (c1 * 2 + c2 + c3) >> 2


def Interp6(c1, c2, c3):
    return ((((c1 & 0x00FF00) * 5 + (c2 & 0x00FF00) * 2 + (c3 & 0x00FF00)) & 0x0007F800) + \
            (((c1 & 0xFF00FF) * 5 + (c2 & 0xFF00FF) * 2 + (c3 & 0xFF00FF)) & 0x07F807F8)) >> 3


def Interp7(c1, c2, c3):
    return ((((c1 & 0x00FF00) * 6 + (c2 & 0x00FF00) + (c3 & 0x00FF00)) & 0x0007F800) + \
            (((c1 & 0xFF00FF) * 6 + (c2 & 0xFF00FF) + (c3 & 0xFF00FF)) & 0x07F807F8)) >> 3


def Interp9(c1, c2, c3):
    return ((((c1 & 0x00FF00) * 2 + ((c2 & 0x00FF00) + (c3 & 0x00FF00)) * 3) & 0x0007F800) + \
            (((c1 & 0xFF00FF) * 2 + ((c2 & 0xFF00FF) + (c3 & 0xFF00FF)) * 3) & 0x07F807F8)) >> 3


def Interp10(c1, c2, c3):
    return ((((c1 & 0x00FF00) * 14 + (c2 & 0x00FF00) + (c3 & 0x00FF00)) & 0x000FF000) +
            (((c1 & 0xFF00FF) * 14 + (c2 & 0xFF00FF) + (c3 & 0xFF00FF)) & 0x0FF00FF0)) >> 4


def PIXEL00_0(rgb_out, pOut, BpL, c): rgb_out[pOut] = c[5]


def PIXEL00_10(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp1(c[5], c[1])


def PIXEL00_11(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp1(c[5], c[4])


def PIXEL00_12(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp1(c[5], c[2])


def PIXEL00_20(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp2(c[5], c[4], c[2])


def PIXEL00_21(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp2(c[5], c[1], c[2])


def PIXEL00_22(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp2(c[5], c[1], c[4])


def PIXEL00_60(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp6(c[5], c[2], c[4])


def PIXEL00_61(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp6(c[5], c[4], c[2])


def PIXEL00_70(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp7(c[5], c[4], c[2])


def PIXEL00_90(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp9(c[5], c[4], c[2])


def PIXEL00_100(rgb_out, pOut, BpL, c): rgb_out[pOut] = Interp10(c[5], c[4], c[2])


def PIXEL01_0(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = c[5]


def PIXEL01_10(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp1(c[5], c[3])


def PIXEL01_11(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp1(c[5], c[2])


def PIXEL01_12(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp1(c[5], c[6])


def PIXEL01_20(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp2(c[5], c[2], c[6])


def PIXEL01_21(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp2(c[5], c[3], c[6])


def PIXEL01_22(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp2(c[5], c[3], c[2])


def PIXEL01_60(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp6(c[5], c[6], c[2])


def PIXEL01_61(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp6(c[5], c[2], c[6])


def PIXEL01_70(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp7(c[5], c[2], c[6])


def PIXEL01_90(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp9(c[5], c[2], c[6])


def PIXEL01_100(rgb_out, pOut, BpL, c): rgb_out[pOut + 1] = Interp10(c[5], c[2], c[6])


def PIXEL10_0(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = c[5]


def PIXEL10_10(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp1(c[5], c[7])


def PIXEL10_11(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp1(c[5], c[8])


def PIXEL10_12(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp1(c[5], c[4])


def PIXEL10_20(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp2(c[5], c[8], c[4])


def PIXEL10_21(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp2(c[5], c[7], c[4])


def PIXEL10_22(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp2(c[5], c[7], c[8])


def PIXEL10_60(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp6(c[5], c[4], c[8])


def PIXEL10_61(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp6(c[5], c[8], c[4])


def PIXEL10_70(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp7(c[5], c[8], c[4])


def PIXEL10_90(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp9(c[5], c[8], c[4])


def PIXEL10_100(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL] = Interp10(c[5], c[8], c[4])


def PIXEL11_0(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = c[5]


def PIXEL11_10(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp1(c[5], c[9])


def PIXEL11_11(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp1(c[5], c[6])


def PIXEL11_12(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp1(c[5], c[8])


def PIXEL11_20(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp2(c[5], c[6], c[8])


def PIXEL11_21(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp2(c[5], c[9], c[8])


def PIXEL11_22(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp2(c[5], c[9], c[6])


def PIXEL11_60(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp6(c[5], c[8], c[6])


def PIXEL11_61(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp6(c[5], c[6], c[8])


def PIXEL11_70(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp7(c[5], c[6], c[8])


def PIXEL11_90(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp9(c[5], c[6], c[8])


def PIXEL11_100(rgb_out, pOut, BpL, c): rgb_out[pOut + BpL + 1] = Interp10(c[5], c[6], c[8])


pattern_dict = dict()


def f0(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[0] = f0
pattern_dict[1] = f0
pattern_dict[4] = f0
pattern_dict[32] = f0
pattern_dict[128] = f0
pattern_dict[5] = f0
pattern_dict[132] = f0
pattern_dict[160] = f0
pattern_dict[33] = f0
pattern_dict[129] = f0
pattern_dict[36] = f0
pattern_dict[133] = f0
pattern_dict[164] = f0
pattern_dict[161] = f0
pattern_dict[37] = f0
pattern_dict[165] = f0

w = 10 * [0]


def f1(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[2] = f1
pattern_dict[34] = f1
pattern_dict[130] = f1
pattern_dict[162] = f1


def f2(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[16] = f2
pattern_dict[17] = f2
pattern_dict[48] = f2
pattern_dict[49] = f2


def f3(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[64] = f3
pattern_dict[65] = f3
pattern_dict[68] = f3
pattern_dict[69] = f3


def f4(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[8] = f4
pattern_dict[12] = f4
pattern_dict[136] = f4
pattern_dict[140] = f4


def f5(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[3] = f5
pattern_dict[35] = f5
pattern_dict[131] = f5
pattern_dict[163] = f5


def f6(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[6] = f6
pattern_dict[38] = f6
pattern_dict[134] = f6
pattern_dict[166] = f6


def f7(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[20] = f7
pattern_dict[21] = f7
pattern_dict[52] = f7
pattern_dict[53] = f7


def f8(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[144] = f8
pattern_dict[145] = f8
pattern_dict[176] = f8
pattern_dict[177] = f8


def f9(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[192] = f9
pattern_dict[193] = f9
pattern_dict[196] = f9
pattern_dict[197] = f9


def f10(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[96] = f10
pattern_dict[97] = f10
pattern_dict[100] = f10
pattern_dict[101] = f10


def f11(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[40] = f11
pattern_dict[44] = f11
pattern_dict[168] = f11
pattern_dict[172] = f11


def f12(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[9] = f12
pattern_dict[13] = f12
pattern_dict[137] = f12
pattern_dict[141] = f12


def f13(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[18] = f13
pattern_dict[50] = f13


def f14(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[80] = f14
pattern_dict[81] = f14


def f15(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[72] = f15
pattern_dict[76] = f15


def f16(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[10] = f16
pattern_dict[138] = f16


def f17(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[66] = f17


def f18(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[24] = f18


def f19(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[7] = f19
pattern_dict[39] = f19
pattern_dict[135] = f19


def f20(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[148] = f20
pattern_dict[149] = f20
pattern_dict[180] = f20


def f21(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[224] = f21
pattern_dict[228] = f21
pattern_dict[225] = f21


def f22(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[41] = f22
pattern_dict[169] = f22
pattern_dict[45] = f22


def f23(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[22] = f23
pattern_dict[54] = f23


def f24(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[208] = f24
pattern_dict[209] = f24


def f25(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[104] = f25
pattern_dict[108] = f25


def f26(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[11] = f26
pattern_dict[139] = f26


def f27(rgb_out, pOut, BpL, c):
    if (diff(w[2], w[6])):
        PIXEL00_11(rgb_out, pOut, BpL, c)
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_60(rgb_out, pOut, BpL, c)
        PIXEL01_90(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[19] = f27
pattern_dict[51] = f27


def f28(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
        PIXEL11_12(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_90(rgb_out, pOut, BpL, c)
        PIXEL11_61(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)


pattern_dict[146] = f28
pattern_dict[178] = f28


def f29(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL01_11(rgb_out, pOut, BpL, c)
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_60(rgb_out, pOut, BpL, c)
        PIXEL11_90(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)


pattern_dict[84] = f29
pattern_dict[85] = f29


def f30(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL10_12(rgb_out, pOut, BpL, c)
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_61(rgb_out, pOut, BpL, c)
        PIXEL11_90(rgb_out, pOut, BpL, c)


pattern_dict[112] = f30
pattern_dict[113] = f30


def f31(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
        PIXEL11_11(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_90(rgb_out, pOut, BpL, c)
        PIXEL11_60(rgb_out, pOut, BpL, c)


pattern_dict[200] = f31
pattern_dict[204] = f31


def f32(rgb_out, pOut, BpL, c):
    if (diff(w[8], w[4])):
        PIXEL00_12(rgb_out, pOut, BpL, c)
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_61(rgb_out, pOut, BpL, c)
        PIXEL10_90(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[73] = f32
pattern_dict[77] = f32


def f33(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
        PIXEL10_11(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_90(rgb_out, pOut, BpL, c)
        PIXEL10_60(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[42] = f33
pattern_dict[170] = f33


def f34(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
        PIXEL01_12(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_90(rgb_out, pOut, BpL, c)
        PIXEL01_61(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[14] = f34
pattern_dict[142] = f34


def f35(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[67] = f35


def f36(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[70] = f36


def f37(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[28] = f37


def f38(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[152] = f38


def f39(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[194] = f39


def f40(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[98] = f40


def f41(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[56] = f41


def f42(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[25] = f42


def f43(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[26] = f43
pattern_dict[31] = f43


def f44(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[82] = f44
pattern_dict[214] = f44


def f45(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[88] = f45
pattern_dict[248] = f45


def f46(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[74] = f46
pattern_dict[107] = f46


def f47(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[27] = f47


def f48(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[86] = f48


def f49(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[216] = f49


def f50(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[106] = f50


def f51(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[30] = f51


def f52(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[210] = f52


def f53(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[120] = f53


def f54(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[75] = f54


def f55(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[29] = f55


def f56(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[198] = f56


def f57(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[184] = f57


def f58(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[99] = f58


def f59(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[57] = f59


def f60(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[71] = f60


def f61(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[156] = f61


def f62(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[226] = f62


def f63(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[60] = f63


def f64(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[195] = f64


def f65(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[102] = f65


def f66(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[153] = f66


def f67(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[58] = f67


def f68(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[83] = f68


def f69(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[92] = f69


def f70(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[202] = f70


def f71(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[78] = f71


def f72(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[154] = f72


def f73(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[114] = f73


def f74(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[89] = f74


def f75(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[90] = f75


def f76(rgb_out, pOut, BpL, c):
    if (diff(w[2], w[6])):
        PIXEL00_11(rgb_out, pOut, BpL, c)
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_60(rgb_out, pOut, BpL, c)
        PIXEL01_90(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[55] = f76
pattern_dict[23] = f76


def f77(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
        PIXEL11_12(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_90(rgb_out, pOut, BpL, c)
        PIXEL11_61(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)


pattern_dict[182] = f77
pattern_dict[150] = f77


def f78(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL01_11(rgb_out, pOut, BpL, c)
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_60(rgb_out, pOut, BpL, c)
        PIXEL11_90(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)


pattern_dict[213] = f78
pattern_dict[212] = f78


def f79(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL10_12(rgb_out, pOut, BpL, c)
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_61(rgb_out, pOut, BpL, c)
        PIXEL11_90(rgb_out, pOut, BpL, c)


pattern_dict[241] = f79
pattern_dict[240] = f79


def f80(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
        PIXEL11_11(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_90(rgb_out, pOut, BpL, c)
        PIXEL11_60(rgb_out, pOut, BpL, c)


pattern_dict[236] = f80
pattern_dict[232] = f80


def f81(rgb_out, pOut, BpL, c):
    if (diff(w[8], w[4])):
        PIXEL00_12(rgb_out, pOut, BpL, c)
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_61(rgb_out, pOut, BpL, c)
        PIXEL10_90(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[109] = f81
pattern_dict[105] = f81


def f82(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
        PIXEL10_11(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_90(rgb_out, pOut, BpL, c)
        PIXEL10_60(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[171] = f82
pattern_dict[43] = f82


def f83(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
        PIXEL01_12(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_90(rgb_out, pOut, BpL, c)
        PIXEL01_61(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[143] = f83
pattern_dict[15] = f83


def f84(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[124] = f84


def f85(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[203] = f85


def f86(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[62] = f86


def f87(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[211] = f87


def f88(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[118] = f88


def f89(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[217] = f89


def f90(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[110] = f90


def f91(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[155] = f91


def f92(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[188] = f92


def f93(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[185] = f93


def f94(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[61] = f94


def f95(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[157] = f95


def f96(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[103] = f96


def f97(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[227] = f97


def f98(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[230] = f98


def f99(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[199] = f99


def f100(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[220] = f100


def f101(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[158] = f101


def f102(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[234] = f102


def f103(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[242] = f103


def f104(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[59] = f104


def f105(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[121] = f105


def f106(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[87] = f106


def f107(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[79] = f107


def f108(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[122] = f108


def f109(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[94] = f109


def f110(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[218] = f110


def f111(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[91] = f111


def f112(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[229] = f112


def f113(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[167] = f113


def f114(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[173] = f114


def f115(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[181] = f115


def f116(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[186] = f116


def f117(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[115] = f117


def f118(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[93] = f118


def f119(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[206] = f119


def f120(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_70(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[205] = f120
pattern_dict[201] = f120


def f121(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_70(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[174] = f121
pattern_dict[46] = f121


def f122(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_70(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[179] = f122
pattern_dict[147] = f122


def f123(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_10(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_70(rgb_out, pOut, BpL, c)


pattern_dict[117] = f123
pattern_dict[116] = f123


def f124(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[189] = f124


def f125(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[231] = f125


def f126(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[126] = f126


def f127(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[219] = f127


def f128(rgb_out, pOut, BpL, c):
    if (diff(w[8], w[4])):
        PIXEL00_12(rgb_out, pOut, BpL, c)
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_61(rgb_out, pOut, BpL, c)
        PIXEL10_90(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[125] = f128


def f129(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL01_11(rgb_out, pOut, BpL, c)
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_60(rgb_out, pOut, BpL, c)
        PIXEL11_90(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)


pattern_dict[221] = f129


def f130(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
        PIXEL01_12(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_90(rgb_out, pOut, BpL, c)
        PIXEL01_61(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[207] = f130


def f131(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
        PIXEL11_11(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_90(rgb_out, pOut, BpL, c)
        PIXEL11_60(rgb_out, pOut, BpL, c)


pattern_dict[238] = f131


def f132(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
        PIXEL11_12(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_90(rgb_out, pOut, BpL, c)
        PIXEL11_61(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)


pattern_dict[190] = f132


def f133(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
        PIXEL10_11(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_90(rgb_out, pOut, BpL, c)
        PIXEL10_60(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[187] = f133


def f134(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL10_12(rgb_out, pOut, BpL, c)
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_61(rgb_out, pOut, BpL, c)
        PIXEL11_90(rgb_out, pOut, BpL, c)


pattern_dict[243] = f134


def f135(rgb_out, pOut, BpL, c):
    if (diff(w[2], w[6])):
        PIXEL00_11(rgb_out, pOut, BpL, c)
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_60(rgb_out, pOut, BpL, c)
        PIXEL01_90(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[119] = f135


def f136(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[237] = f136
pattern_dict[233] = f136


def f137(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[175] = f137
pattern_dict[47] = f137


def f138(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[183] = f138
pattern_dict[151] = f138


def f139(rgb_out, pOut, BpL, c):
    PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[245] = f139
pattern_dict[244] = f139


def f140(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[250] = f140


def f141(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[123] = f141


def f142(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[95] = f142


def f143(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[222] = f143


def f144(rgb_out, pOut, BpL, c):
    PIXEL00_21(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[252] = f144


def f145(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_22(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[249] = f145


def f146(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_21(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[235] = f146


def f147(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_22(rgb_out, pOut, BpL, c)


pattern_dict[111] = f147


def f148(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_21(rgb_out, pOut, BpL, c)


pattern_dict[63] = f148


def f149(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    PIXEL10_22(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[159] = f149


def f150(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    PIXEL10_21(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[215] = f150


def f151(rgb_out, pOut, BpL, c):
    PIXEL00_22(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[246] = f151


def f152(rgb_out, pOut, BpL, c):
    PIXEL00_10(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[254] = f152


def f153(rgb_out, pOut, BpL, c):
    PIXEL00_12(rgb_out, pOut, BpL, c)
    PIXEL01_11(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[253] = f153


def f154(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    PIXEL01_10(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[251] = f154


def f155(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    PIXEL01_12(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    PIXEL11_11(rgb_out, pOut, BpL, c)


pattern_dict[239] = f155


def f156(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_20(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_20(rgb_out, pOut, BpL, c)
    PIXEL11_10(rgb_out, pOut, BpL, c)


pattern_dict[127] = f156


def f157(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    PIXEL10_11(rgb_out, pOut, BpL, c)
    PIXEL11_12(rgb_out, pOut, BpL, c)


pattern_dict[191] = f157


def f158(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_20(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    PIXEL10_10(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_20(rgb_out, pOut, BpL, c)


pattern_dict[223] = f158


def f159(rgb_out, pOut, BpL, c):
    PIXEL00_11(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    PIXEL10_12(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[247] = f159


def f160(rgb_out, pOut, BpL, c):
    if (diff(w[4], w[2])):
        PIXEL00_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL00_100(rgb_out, pOut, BpL, c)
    if (diff(w[2], w[6])):
        PIXEL01_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL01_100(rgb_out, pOut, BpL, c)
    if (diff(w[8], w[4])):
        PIXEL10_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL10_100(rgb_out, pOut, BpL, c)
    if (diff(w[6], w[8])):
        PIXEL11_0(rgb_out, pOut, BpL, c)
    else:
        PIXEL11_100(rgb_out, pOut, BpL, c)


pattern_dict[255] = f160


def hq2x(xres, yres, rgb):
    '''
    +--+--+--+
    |w1|w2|w3|
    +--+--+--+
    |w4|w5|w6|
    +--+--+--+
    |w7|w8|w9|
    +--+--+--+
    '''
    c = 10 * [0]

    rgb_out = 4 * len(rgb) * [0]
    BpL = 2 * xres

    for j in range(yres):
        prevline = -xres if j > 0 else 0
        nextline = xres if j < yres - 1 else 0

        for i in range(xres):
            pos = j * xres + i
            pOut = j * xres * 4 + 2 * i

            w[1] = w[2] = w[3] = rgb[pos + prevline]
            w[4] = w[5] = w[6] = rgb[pos]
            w[7] = w[8] = w[9] = rgb[pos + nextline]

            if i > 0:
                w[1] = rgb[pos + prevline - 1]
                w[4] = rgb[pos - 1]
                w[7] = rgb[pos + nextline - 1]

            if i < xres - 1:
                w[3] = rgb[pos + prevline + 1]
                w[6] = rgb[pos + 1]
                w[9] = rgb[pos + nextline + 1]

            pattern = 0
            flag = 1
            YUV1 = RGBtoYUV[w[5]]
            for k in range(1, 10):
                if k == 5:
                    continue
                if w[k] != w[5]:
                    YUV2 = RGBtoYUV[w[k]]
                    if (abs((YUV1 & Ymask) - (YUV2 & Ymask)) > trY) or \
                            (abs((YUV1 & Umask) - (YUV2 & Umask)) > trU) or \
                            (abs((YUV1 & Vmask) - (YUV2 & Vmask)) > trV):
                        pattern |= flag
                flag <<= 1

            for k in range(1, 10):
                c[k] = LUT16to32[w[k]]

            try:
                f = pattern_dict[pattern]
                f(rgb_out, pOut, BpL, c)
            except KeyError:
                pass

    return rgb_out


def init_LUTs():
    global LUT16to32, RGBtoYUV

    for i in range(65536):
        LUT16to32[i] = ((i & 0xF800) << 8) | ((i & 0x07E0) << 5) | ((i & 0x001F) << 3)

    for i in range(32):
        for j in range(64):
            for k in range(32):
                r = i << 3
                g = j << 2
                b = k << 3
                Y = (r + g + b) >> 2
                u = 128 + ((r - b) >> 2)
                v = 128 + ((-r + 2 * g - b) >> 3)
                RGBtoYUV[(i << 11) | (j << 5) | k] = (Y << 16) | (u << 8) | v


def main():
    init_LUTs()
    # print 'scaling randam.ppm to randam2.ppm (100 times)..'
    ppm = PPM.load(Relative('randam.ppm'))
    for i in range(100):
        rgb = hq2x(ppm.w, ppm.h, ppm.rgb)
    PPM(2 * ppm.w, 2 * ppm.h, rgb).save(Relative('randam2.ppm'))


def run():
    main()
    return True


run()
