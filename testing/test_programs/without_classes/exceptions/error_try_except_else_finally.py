try:
    a = 3
except KeyError as k:
    a = "3"
except Exception as e:
    a = list()
else:
    a = dict()
finally:
    a = 3.2

r1 = len(a) # Detected. Using finally is checked, as the type of a is float