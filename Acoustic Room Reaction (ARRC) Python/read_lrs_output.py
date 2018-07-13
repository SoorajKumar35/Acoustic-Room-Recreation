from subprocess import call
from fractions import Fraction

def read_lrs_output(filename):

    V = []
    f = open(filename, 'r')
    for line in f:
        line = line.rstrip('\n')
        if(line == 'end'):
            f.close()
            return V
        if(len(line) > 0):
            if(line[1] == '1'):
                row = []
                line_list = line.split()
                for item in line_list:
                    value = get_float(item)
                    row += [value]
                V.append(row)

def get_float(rat):
    if(rat == '1'):
        return 1
    else:
        numer = ''
        #Check for backslash
        if '/' not in rat:
            return float(rat)
        else:
            split_list = rat.split('/')
            value = float(split_list[0])/float(split_list[1])
            return value

def write_input(A, b, filename):
    f = open(filename, 'w')
    f.write('cube\n')
    f.write('H-representation\n')
    f.write('begin\n')
    num_con = len(A)
    vars = 4
    string_one = str(num_con) + ' ' + str(vars) + ' ' + 'rational\n'
    f.write(string_one)
    for i in range(len(A)):
        b_i = ''
        A_i0 = ''
        A_i1 = ''
        A_i2 = ''

        if isinstance(b[i],float):
            b_i = str(Fraction(b[i]).limit_denominator())
        else:
            b_i = str(b[i])
        if isinstance(A[i][0],float):
            A_i0 = str(Fraction(-A[i][0]).limit_denominator())
        else:
            A_i0 = str(-A[i][0])
        if isinstance(A[i][1],float):
            A_i1 = str(Fraction(-A[i][1]).limit_denominator())
        else:
            A_i1 = str(-A[i][1])
        if isinstance(A[i][2],float):
            A_i2 = str(Fraction(-A[i][2]).limit_denominator())
        else:
            A_i2 = str(-A[i][2])

        string_two = b_i + ' ' + A_i0 + ' ' + A_i1 + ' ' + A_i2 + '\n'
        f.write(string_two)
    f.write('end')
    f.close()

def call_lrs(A,b):
    write_input(A,b,'input.txt')
    f = open('output.txt', 'w')
    call(['./lrs', 'input.txt'], stdout=f)
    return read_lrs_output('output.txt')

# read_lrs_output('output.txt')
