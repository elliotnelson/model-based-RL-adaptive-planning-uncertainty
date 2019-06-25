import decimal

# copied from stackoverflow.com/questions/38847690/convert-float-to-string-without-scientific-notation-and-false-precision

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def log_list_to_file(l, filename):

    f = open(filename,'a')
    f.write('{')
    f.write(','.join([float_to_str(x) for x in l]))
    f.write('},') # note: removed \n from argument because Mathematica interprets it as an extra comma, apparently
    f.close()

    return

