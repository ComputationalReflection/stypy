
try:
    # a: int
    a = 3
except KeyError as k:
    # a: str
    a = '3'
except Exception as e:
    # a: list[]
    a = list()
else:
    # a: dict{}
    a = dict()