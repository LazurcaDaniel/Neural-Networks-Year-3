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
                
                
def determinant(A):
    """
    This function will return the determinant of a matrix
    """
    l1 = A[1][1] * (A[2][2] * A[3][3] - A[2][3] * A[3][2])
    l2 = A[1][2] * (A[2][1] * A[3][3] - A[2][3] * A[3][1])
    l3 = A[1][3] * (A[2][1] * A[3][2] - A[2][2] * A[3][1])
    return l1 - l2 + l3

def trace(A):
    """
    This function will return the trace of a matrix
    """
    return A[1][1] + A[2][2] + A[3][3]

def vector_norm(B):
    """
    This function will return the norm of a vector
    """
    return (B[1]*B[1] + B[2]*B[2] + B[3]*B[3])**0.5

def transpose(A):
    """
    This function will return the transpose of a matrix
    """
    B=[]
    for i in range(len(A[0])):
        new_line = []
        for j in range(len(A)):
            new_line.append(A[j][i])
        B.append(new_line)
    return B

def multiply_with_vector(A,B):
    """
    This function will return the result of multiplying the matrix A with vector B
    """
    result = []
    for i in range(len(A)):
        el = 0
        for j in range(len(A[i])):
            el = el + A[i][j] * B[j]
        result.append(el)
    return result




(A,B) = readFromFile('test.txt')
print(multiply_with_vector(A,B))