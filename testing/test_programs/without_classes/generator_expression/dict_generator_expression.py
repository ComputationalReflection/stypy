import six

defaultParams = {0: ('zero', '0'), 1: ('one', '1')}

param1 = six.iteritems(defaultParams)

validate = dict((key, converter) for key, (default, converter) in six.iteritems(defaultParams))


params2 = {0: '0', 1: '1'}

it = six.iteritems(params2)


