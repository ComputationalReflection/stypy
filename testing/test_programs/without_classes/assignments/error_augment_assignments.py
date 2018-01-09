l = [1, 2, 4, 5]
l[0] = l[0] - "a"  # Error detected
l[0] -= "a"  # Not detected

s = 3
s = s + str(3)  # Error detected

s += str(3)  # Not detected
s += str(5)
s += str(7)
