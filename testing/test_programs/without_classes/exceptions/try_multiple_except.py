try:
    a = 3
except KeyError as k:
    a = "3"
except Exception as e:
    a = list()
