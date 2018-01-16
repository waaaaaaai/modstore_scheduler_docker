
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


# http://stackoverflow.com/questions/32235351/python-decorator-to-print-return-values
def monitor_results(func):
    def wrapper(*func_args, **func_kwargs):
        retval = func(*func_args,**func_kwargs) #caller
        print('> ' + func.__name__ + '({}, {}) = {}'.format(func_args[1:], func_kwargs, repr(retval)))
        # print('function ' + func.__name__ + '() returns ' + repr(retval))
        return retval
    wrapper.__name__ = func.__name__
    return wrapper


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def float_leq(a, b):  # less or equal than for float number
    return a < b or isclose(a, b)


def float_geq(a, b):
    return a > b or isclose(a, b)
