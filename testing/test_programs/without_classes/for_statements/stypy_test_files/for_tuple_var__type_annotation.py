
# arguments: dict[{str: int}]
arguments = {'a': 1, 'b': 2}
# ret_str: str
ret_str = ''
# key: str \/ int; arg: str \/ int
for (key, arg) in arguments.items():
    # ret_str: str
    ret_str += ((str(key) + ': ') + str(arg))
print ret_str