import sympy as sp

def solve_quadratic(a, b, c):
    x = sp.Symbol('x')
    eq = a*x**2 + b*x + c
    sol = sp.solve(eq, x)
    return sol

# Sample use
if __name__ == "__main__":
    print(solve_quadratic(1, -3, 2))  # [1, 2]
