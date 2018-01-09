glob = 0

class _ctypes(object):
    att1 = glob
    att2 = att1

    def data_as(self, obj):
        return 0

    def shape_as(self, obj):
        return 1

    def strides_as(self, obj):
        return 2

    def get_data(self):
        return 3

    def get_shape(self):
        return 4

    def get_strides(self):
        return 5

    def get_as_parameter(self):
        return 6

    data = property(get_data, None, doc="c-types data")
    shape = property(get_shape, None, doc="c-types shape")
    strides = property(get_strides, None, doc="c-types strides")
    _as_parameter_ = property(get_as_parameter, None, doc="_as parameter_")

ct = _ctypes()

r = ct.data
r2 = ct.shape
r3 = ct.strides
r4 = ct.get_data

ra = ct.att1
rb = ct.att2
