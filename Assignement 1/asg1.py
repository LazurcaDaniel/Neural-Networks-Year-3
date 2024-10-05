def readFromFile(filename):
    """
    This function will read a system of linear  equations from a text file
    and return a matrix A of qoefficients and a vector b of constants.
    """
    A = []
    B = []
    with open(filename, 'r') as f:
        for n in f:
            n = n.split()
            n_line = []
            pozitive = True
            for i in n:
                if(i == '='):
                    pozitive = True
                    continue
                elif(i.endswith('x') or i.endswith('y') or i.endswith('z')):
                    if(len(i) == 1):
                        n_line.append(1)
                    else:
                        n_line.append(int(i[:-1]))
                    if(pozitive == False):
                        n_line[-1] = -n_line[-1]
                elif(i.isnumeric()):
                    B.append(int(i))
                    if(pozitive == False):
                        B[-1] = -B[-1]
                elif(i == '-'):
                    pozitive = False
                elif(i == '+'):
                    pozitive = True
            A.append(n_line)
    return A, B
                
                


(A,B) = readFromFile('test.txt')
