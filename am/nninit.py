import sys

def dump_vec(vec):
    sys.stdout.write('(')
    sys.stdout.write('[{}], '.format(len(vec)))
    sys.stdout.write('[{}]'.format(', '.join(['{:.6}'.format(e) for e in vec])))
    sys.stdout.write(')\n')

def dump_mat(mat):
    sys.stdout.write('(')
    sys.stdout.write('[{}, {}], '.format(len(mat), len(mat[0])))
    sys.stdout.write('[')
    for i, v in enumerate(mat):
        sys.stdout.write(', '.join(['{:.6}'.format(e) for e in v]))

        if i != len(mat) - 1:
            sys.stdout.write(', ')
    sys.stdout.write(']')
    sys.stdout.write(')\n')

def gen_mat(m, n, gen):
    w = []
    for i in range(m):
        v = []
        for j in range(n):
            v.append(gen(n + m))
        w.append(v)
    dump_mat(w)

def gen_vec(n, gen):
    v = []
    for j in range(n):
        v.append(gen(n))
    dump_vec(v)

def dump_tensor(sizes, vec):
    sys.stdout.write('(')
    sys.stdout.write('[{}]'.format(', '.join([str(s) for s in sizes])))
    sys.stdout.write(', ')
    sys.stdout.write('[{}]'.format(', '.join(['{:.6}'.format(e) for e in vec])))
    sys.stdout.write(')\n')

def gen_tensor(sizes, gen):
    d = 0 if len(sizes) == 0 else 1

    for s in sizes:
        d *= s

    result = []
    for i in range(d):
        result.append(gen(d / sizes[0]))

    dump_tensor(sizes, result)

