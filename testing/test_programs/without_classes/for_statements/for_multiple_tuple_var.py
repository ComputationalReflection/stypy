
kwargs = {
    'constant_values': [(1, 1.0),  (2, 2.0), (3, 3.0)]
}
pad_width = [('a', 'a'), ('b', 'b'), ('c', 'c')]

for axis, ((pad_before, pad_after), (before_val, after_val)) in enumerate(
        zip(pad_width, kwargs['constant_values'])):
    print axis
    print pad_before
    print pad_after
    print before_val
    print after_val