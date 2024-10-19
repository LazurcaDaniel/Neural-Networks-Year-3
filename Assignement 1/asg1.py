import copy

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
    This function will return the determinant of a matrix of degree 3
    """
    l1 = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
    l2 = A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
    l3 = A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    return l1 - l2 + l3

def determinant_2(A):
    """
    This function will return the determinant of a matrix of degree 2
    """
    l1 = A[0][0] * A[1][1]
    l2 = A[1][0] * A[0][1]
    return l1-l2


def trace(A):
    """
    This function will return the trace of a matrix
    """
    return A[0][0] + A[1][1] + A[2][2]

def vector_norm(B):
    """
    This function will return the norm of a vector
    """
    return (B[0]*B[0] + B[1]*B[1] + B[2]*B[2])**0.5

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

def determinant_Ai(A, B, variable):
    """
    This function will return the determinant of Ax, Ay or Az, depending on variable value
    0 for x
    1 for y
    2 for z
    """
    A_copy = copy.deepcopy(A)
    for i in range(len(A)):
        A_copy[i][variable] = B[i]

    return determinant(A_copy)


def Cramer(A, B):
    """
    This function will solve a system of polynomials usign Cramer's Rule
    """
    x = determinant_Ai(A,B,0)/determinant(A)
    y = determinant_Ai(A,B,1)/determinant(A)
    z = determinant_Ai(A,B,2)/determinant(A)
    return (x,y,z)

def cofactor(A):
    """
    This function will return the cofactor matrix of matrix A
    """
    cof = []
    for i in range(len(A)):
        line_i = []
        for j in range(len(A[i])):
            M = []
            for k in range(len(A)):
                new_line = []
                if k != i:
                    for l in range(len(A[k])):
                        if l != j:
                            new_line.append(A[k][l])
                    M.append(new_line)
            line_i.append(((-1)**(i+j)) * determinant_2(M))
            
        cof.append(line_i)
    return cof

def inverse_matrix(A):
    """
    This function will return A^-1, using the adjucate method
    """
    A1 = transpose(cofactor(A))
    inverse = 1/determinant(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            A1[i][j] = inverse * A1[i][j]
    return A1

(A,B) = readFromFile(r'C:\Users\Daniel\Desktop\RN\Assignement 1\test.txt')
print(Cramer(A,B))
print(multiply_with_vector(inverse_matrix(A),B))