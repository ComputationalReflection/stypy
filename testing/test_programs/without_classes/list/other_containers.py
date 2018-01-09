


import os

x = os.environ

r = x['COMPUTERNAME']

x['FOO'] = 'BAR'

r2 = x['FOO']
