import tkinter as tk
from tkinter import messagebox
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
import sympy as sp
from sympy import (
    symbols, simplify, solve, factor, Eq, expand, Abs, Piecewise, sympify,solveset, S,
    sin, cos, tan, csc, sec, cot, sqrt, pi, Poly, lambdify, Mod, divisors,solve_univariate_inequality,diff,limit, gcd,
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.sets import Interval
import traceback
from sympy.core.sympify import SympifyError
import logging

# Configure logging
logging.basicConfig(filename='calculator_errors.log', level=logging.ERROR)

# Define symbols 'x' and 'h' for algebraic operations
x = symbols('x')

# ------------------- Utility Functions ------------------- #
def format_solution(value):
    """Returns a string representing the number with up to 4 decimal places."""
    try:
        # If value is a SymPy object, convert it to float
        if isinstance(value, sympy.Basic):
            value = float(value.evalf())
        # Return as string with up to 4 decimal places
        return f"{value:.4f}"
    except Exception as e:
        raise ValueError(f"Error in format_solution: {str(e)}")

def add_multiplication_sign(expression):
    """Insert explicit multiplication signs where needed, replace '^' with '**'."""
    # Replace '^' with '**' for exponentiation
    expression = expression.replace('^', '**')
    # Insert '*' between a number and a variable (e.g., '2x' -> '2*x')
    expression = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expression)
    # Insert '*' between closing and opening parentheses (e.g., ')( -> ')*(')
    expression = re.sub(r'(\))(\()', r'\1*\2', expression)
    return expression

def validate_expression(expr_str):
    """
    Validates and formats the expression string for SymPy.
    """
    try:
        formatted_expr = add_multiplication_sign(expr_str.replace('^', '**').replace(' ', ''))
        sympy_expr = sympify(formatted_expr)
        return sympy_expr
    except Exception as e:
        raise ValueError(f"Invalid expression '{expr_str}': {str(e)}")

def validate_operator(op):
    """
    Validates the inequality operator.
    """
    if op not in [">", "<", ">=", "<="]:
        raise ValueError(f"Invalid operator '{op}'. Choose from '>', '<', '>=', '<='.")
    return op

def validate_numeric(value):
    """
    Validates that the input value is numeric.
    """
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{value}'. Please enter a valid number.")

def is_whole_number(value, tol=1e-5):
    """Check if the value is a whole number within a small tolerance."""
    return np.isclose(value, round(value), atol=tol)

def safe_fraction_or_float(value):
    """Attempts to convert a value to a Fraction, or fall back to float if necessary."""
    try:
        return Fraction(value)
    except ValueError:
        # If it's not a valid fraction, try converting it to a float
        return float(value)

def format_interval(interval):
    """
    Formats a SymPy Interval object into a string with appropriate brackets.

    Parameters:
    - interval: sympy.Interval

    Returns:
    - str: Formatted interval string
    """
    # Determine left bracket
    if interval.left_open:
        left_bracket = '('
    else:
        left_bracket = '['

    # Determine right bracket
    if interval.right_open:
        right_bracket = ')'
    else:
        right_bracket = ']'

    # Format left bound
    if interval.start == -sympy.oo:
        left_bound = '-∞'
    else:
        left_bound = str(interval.start)

    # Format right bound
    if interval.end == sympy.oo:
        right_bound = '∞'
    else:
        right_bound = str(interval.end)

    return f"{left_bracket}{left_bound}, {right_bound}{right_bracket}"

def is_parentheses_balanced(expr):
    stack = []
    for char in expr:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0


# ------------------- Coefficient Extraction Functions ------------------- #
def extract_coefficients(equation):
    """
    Extracts coefficients from standard and slope-intercept form equations.
    Supports equations of the form:
    - Ax + By = C
    - By + Ax = C
    - y = mx + b
    """
    equation = equation.replace(" ", "").lower()
    match_standard = re.match(r"([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)y=([-+]?\d+\.?\d*)", equation)
    match_reverse_standard = re.match(r"([-+]?\d*\.?\d*)y([-+]?\d*\.?\d*)x=([-+]?\d+\.?\d*)", equation)
    match_slope_intercept = re.match(r"y=([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)?", equation)

    if match_standard:
        A, B, C = match_standard.groups()
    elif match_reverse_standard:
        B, A, C = match_reverse_standard.groups()
    elif match_slope_intercept:
        m, b = match_slope_intercept.groups()
        A, B, C = -float(m or "1"), 1.0, float(b or "0")
    else:
        raise ValueError("Invalid equation format. Use 'Ax + By = C' or 'y = mx + b'.")

    # Handle cases where coefficients are missing (implying 1 or -1)
    A = float(A) if A not in ["", "+", "-"] else float(f"{A}1")
    B = float(B) if B not in ["", "+", "-"] else float(f"{B}1")
    C = float(C)
    return A, B, C

def extract_coefficients_3x3(equation):
    """
    Extracts coefficients from 3-variable standard form equations.
    Supports equations of the form:
    - Ax + By + Cz = D
    """
    equation = equation.replace(" ", "").lower()
    match = re.match(r"([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)y([-+]?\d*\.?\d*)z=([-+]?\d+\.?\d*)", equation)

    if match:
        A, B, C, D = match.groups()
        A = float(A) if A not in ["", "+", "-"] else float(f"{A}1")
        B = float(B) if B not in ["", "+", "-"] else float(f"{B}1")
        C = float(C) if C not in ["", "+", "-"] else float(f"{C}1")
        D = float(D)
        return A, B, C, D
    else:
        raise ValueError("Invalid 3-variable equation format. Use 'Ax + By + Cz = D'.")

# ------------------- Equation Operations ------------------- #
def calculate_slope_and_intercept(equation):
    """Calculates slope and y-intercept from an equation."""
    A, B, C = extract_coefficients(equation)
    if B == 0:
        slope = "Undefined (Vertical line)"
        y_intercept = None
        x_intercept = C / A if A != 0 else None
    else:
        slope = -A / B
        y_intercept = C / B
        x_intercept = C / A if A != 0 else None

    plot_line(slope, y_intercept, x_intercept)
    return slope, y_intercept

def calculate_slope_from_points(x1, y1, x2, y2):
    """Calculates slope and y-intercept from two points."""
    if x1 == x2:
        slope = "Undefined (Vertical line)"
        y_intercept = None
        x_intercept = x1
        plot_line(slope, y_intercept, x_intercept)
        return slope, y_intercept
    else:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
        x_intercept = -y_intercept / slope if slope != 0 else None
        plot_line(slope, y_intercept, x_intercept)
        return slope, y_intercept

def calculate_slope_for_parallel_perpendicular(equation, x, y, relationship):
    """Calculates slope for parallel or perpendicular lines."""
    A, B, C = extract_coefficients(equation)
    if B == 0:
        original_slope = "Undefined (Vertical line)"
    else:
        original_slope = -A / B

    if relationship == "parallel":
        slope = original_slope
    elif relationship == "perpendicular":
        if original_slope == "Undefined (Vertical line)":
            slope = 0  # Perpendicular to vertical is horizontal
        elif original_slope == 0:
            slope = "Undefined (Vertical line)"  # Perpendicular to horizontal is vertical
        else:
            slope = -1 / original_slope
    else:
        raise ValueError("Invalid relationship. Choose 'parallel' or 'perpendicular'.")

    if slope == "Undefined (Vertical line)":
        y_intercept = None
        x_intercept = x
    else:
        y_intercept = y - slope * x
        x_intercept = -y_intercept / slope if slope != 0 else None

    plot_line(slope, y_intercept, x_intercept)
    return slope, y_intercept

def analyze_quadratic(poly_expr):
    """
    Given a quadratic expression, compute domain, range, intervals of increase/decrease,
    concavity, axis of symmetry, vertex, and y-intercept.
    """
    # Domain: all real numbers
    domain = "All real numbers"

    # Ensure the expression is a SymPy expression
    if not isinstance(poly_expr, sympy.Expr):
        raise ValueError("poly_expr must be a SymPy expression.")

    # Create a polynomial object
    poly = Poly(poly_expr, x)
    coeffs = poly.all_coeffs()

    if len(coeffs) != 3:
        raise ValueError("The polynomial is not quadratic.")

    a, b, c = coeffs

    # Determine concavity
    if a > 0:
        concavity = "Upward"
    elif a < 0:
        concavity = "Downward"
    else:
        raise ValueError("Coefficient 'a' cannot be zero for a quadratic function.")

    # Axis of symmetry
    h = -b / (2 * a)
    axis_of_symmetry = f"x = {format_solution(h)}"

    # Vertex
    k = a*h**2 + b*h + c
    vertex = (format_solution(h), format_solution(k))

    # Y-intercept
    y_intercept = format_solution(c)

    # Range
    if concavity == "Upward":
        range_ = f"[{format_solution(k)}, ∞)"
    else:
        range_ = f"(-∞, {format_solution(k)}]"

    # Find critical points for increasing/decreasing intervals
    derivative = diff(poly_expr, x)
    critical_points = solve(derivative, x)

    # For quadratic, only one critical point, which is the vertex
    if len(critical_points) == 1:
        critical_point = critical_points[0]
        try:
            h_val = float(critical_point.evalf())
        except Exception as e:
            raise ValueError(f"Cannot convert critical point to float: {e}")

        if concavity == "Upward":
            increasing_interval = f"({h_val}, ∞)"
            decreasing_interval = f"(-∞, {h_val})"
        else:
            increasing_interval = f"(-∞, {h_val})"
            decreasing_interval = f"({h_val}, ∞)"
    else:
        increasing_interval = "Undefined"
        decreasing_interval = "Undefined"

    # Find x-intercepts
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        x1 = (-b + sympy.sqrt(discriminant)) / (2*a)
        x2 = (-b - sympy.sqrt(discriminant)) / (2*a)
        try:
            x_intercepts = [format_solution(x1), format_solution(x2)]
        except Exception as e:
            raise ValueError(f"Error formatting x-intercepts: {e}")
    elif discriminant == 0:
        x_root = -b / (2*a)
        try:
            x_intercepts = [format_solution(x_root)]
        except Exception as e:
            raise ValueError(f"Error formatting x-intercept: {e}")
    else:
        x_intercepts = []  # No real roots

    return {
        "domain": domain,
        "range": range_,
        "concavity": concavity,
        "axis_of_symmetry": axis_of_symmetry,
        "vertex": vertex,
        "y_intercept": y_intercept,
        "increasing_intervals": increasing_interval,
        "decreasing_intervals": decreasing_interval,
        "x_intercepts": x_intercepts
    }

def analyze_rational_function(function_str):
    """
    Analyzes a rational function to find asymptotes, intercepts, and holes.

    Parameters:
    - function_str: str, the rational function (e.g., "x/(x*(x+8))")

    Returns:
    - dict containing asymptotes, intercepts, and holes
    """
    try:
        # Validate parentheses
        if not is_parentheses_balanced(function_str):
            raise ValueError("Unbalanced parentheses detected in the expression.")

        # Enable implicit multiplication (allows inputs like 'x/(x(x+8))')
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(function_str, transformations=transformations)

        if not isinstance(expr, sympy.Expr):
            raise ValueError("The input is not a valid expression.")

        # Ensure the function is rational: expression is a ratio of polynomials
        numerator, denominator = expr.as_numer_denom()

        # Attempt to create Poly objects to verify if they are polynomials
        try:
            poly_num = Poly(numerator, x)
            poly_den = Poly(denominator, x)
        except PolynomialError:
            raise ValueError("The function is not a rational function (ratio of polynomials).")

        # Factor numerator and denominator
        numerator_factors = sympy.factor_list(numerator)[1]
        denominator_factors = sympy.factor_list(denominator)[1]

        # Find GCD of numerator and denominator to identify holes
        gcd_expr = sympy.gcd(numerator, denominator)

        # Holes: solutions to gcd_expr = 0
        holes = solveset(gcd_expr, x, domain=S.Reals)
        hole_xs = [float(sol.evalf()) for sol in holes]

        # Vertical Asymptotes: zeros of denominator not canceled by numerator
        vertical_asymptotes_set = solveset(denominator, x, domain=S.Reals) - solveset(gcd_expr, x, domain=S.Reals)
        vertical_asymptotes = [float(sol.evalf()) for sol in vertical_asymptotes_set]

        # Horizontal or Oblique Asymptotes
        deg_num = sympy.degree(numerator, gen=x)
        deg_den = sympy.degree(denominator, gen=x)

        if deg_num < deg_den:
            horizontal_asymptote = limit(expr, x, sympy.oo)
            horizontal_asymptote_exists = True
            oblique_asymptote = None
        elif deg_num == deg_den:
            horizontal_asymptote = limit(expr, x, sympy.oo)
            horizontal_asymptote_exists = True
            oblique_asymptote = None
        else:
            # Oblique asymptote exists
            quotient, remainder = sympy.div(numerator, denominator, domain='EX')
            oblique_asymptote = quotient
            horizontal_asymptote_exists = False
            horizontal_asymptote = None

        # X-Intercepts: zeros of numerator, excluding holes
        x_intercepts_set = solveset(numerator, x, domain=S.Reals) - solveset(gcd_expr, x, domain=S.Reals)
        x_intercepts = [float(sol.evalf()) for sol in x_intercepts_set]

        # Y-Intercept: f(0) if denominator doesn't equal zero
        if denominator.subs(x, 0) != 0:
            y_intercept = expr.subs(x, 0)
            y_intercept_val = float(y_intercept.evalf())
        else:
            y_intercept = None  # Undefined
            y_intercept_val = None

        # Holes points: (a, limit expr as x approaches a)
        hole_points = []
        for a in hole_xs:
            y = limit(expr, x, a)
            if y.is_real:
                hole_points.append( (a, float(y.evalf())) )
            else:
                hole_points.append( (a, None) )

        return {
            "vertical_asymptotes": vertical_asymptotes,
            "horizontal_asymptote": float(horizontal_asymptote.evalf()) if horizontal_asymptote_exists else None,
            "oblique_asymptote": oblique_asymptote if not horizontal_asymptote_exists else None,
            "x_intercepts": x_intercepts,
            "y_intercept": y_intercept_val,
            "holes": hole_points
        }
    except SympifyError:
        raise ValueError("Invalid mathematical expression. Please check your syntax and ensure all parentheses are balanced.")
    except Exception as e:
        logging.error("Error in analyzing rational function", exc_info=True)
        raise ValueError(f"Error in analyzing rational function: {str(e)}")

# ------------------- Plotting Functions ------------------- #
def plot_line(slope, y_intercept, x_intercept):
    """Plots the line given slope and y-intercept."""
    plt.figure(figsize=(6, 6))
    x_vals = np.linspace(-10, 10, 400)

    if slope == "Undefined (Vertical line)":
        x = np.full_like(x_vals, x_intercept)
        y = x_vals
        plt.plot(x, y, '-r', label=f'x = {x_intercept}')
    else:
        y_vals = slope * x_vals + y_intercept
        plt.plot(x_vals, y_vals, '-r', label=f'y = {slope:.2f}x + {y_intercept:.2f}')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def plot_quadratic(expr, quadratic_analysis):
    """
    Plots the quadratic function and marks key features.

    Parameters:
    - expr: sympy expression of the quadratic function
    - quadratic_analysis: dict containing analysis results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import lambdify

    # Create a numerical function from the symbolic expression
    f = lambdify(x, expr, modules=['numpy'])

    # Extract necessary values from quadratic_analysis
    vertex = quadratic_analysis['vertex']
    axis_of_symmetry = quadratic_analysis['axis_of_symmetry']
    concavity = quadratic_analysis['concavity']
    x_intercepts = quadratic_analysis['x_intercepts']
    y_intercept = quadratic_analysis['y_intercept']
    domain = quadratic_analysis['domain']
    range_ = quadratic_analysis['range']
    increasing_intervals = quadratic_analysis['increasing_intervals']
    decreasing_intervals = quadratic_analysis['decreasing_intervals']

    # Generate x values around the vertex for better visualization
    x_min = float(vertex[0]) - 10
    x_max = float(vertex[0]) + 10
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='Quadratic Function', color='blue')

    # Plot vertex
    plt.plot(float(vertex[0]), float(vertex[1]), 'ro', label='Vertex')

    # Plot axis of symmetry
    plt.axvline(float(axis_of_symmetry), color='green', linestyle='--', label='Axis of Symmetry')

    # Plot x-intercepts
    plotted_roots = False
    if x_intercepts:
        for xi in x_intercepts:
            plt.plot(float(xi), 0, 'bo', label='X-Intercept' if not plotted_roots else "")
            plotted_roots = True  # Only label the first intercept

    # Plot y-intercept
    plt.plot(0, float(y_intercept), 'go', label='Y-Intercept')

    # Avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title('Graph of the Quadratic Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.xlim(x_min, x_max)

    # Display domain and range on the plot
    plt.text(0.05, 0.95, f"Domain: {domain}\nRange: {range_}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')

    plt.show()

def plot_polynomial(polynomial_expr, solutions):
    """Plots the polynomial function and marks its roots."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import lambdify

    try:
        # Create a numerical function from the symbolic expression
        f = lambdify(x, polynomial_expr, modules=['numpy'])

        # Extract real solutions
        real_solutions = []
        for sol in solutions:
            if sol.is_real:
                try:
                    sol_val = float(sol.evalf())
                    real_solutions.append(sol_val)
                except Exception as e:
                    print(f"Cannot convert solution {sol} to float: {e}")

        # Determine plotting range
        if real_solutions:
            min_x = min(real_solutions) - 5
            max_x = max(real_solutions) + 5
        else:
            # If no real roots, use a default range
            min_x = -10
            max_x = 10

        x_vals = np.linspace(min_x, max_x, 400)
        y_vals = f(x_vals)

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label='Polynomial Function', color='blue')

        # Mark the real roots without duplicating labels
        plotted_roots = False
        for sol in solutions:
            if sol.is_real:
                try:
                    sol_float = float(sol.evalf())
                    plt.plot(sol_float, 0, 'ro', label='Root' if not plotted_roots else "")
                    plotted_roots = True  # Only label the first root
                except Exception as e:
                    print(f"Cannot plot solution {sol}: {e}")

        plt.title('Graph of the Polynomial Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()
    except Exception as e:
        raise ValueError(f"Error in plot_polynomial: {str(e)}")

def plot_rational_function(function_str, analysis_results):
    """
    Plots the rational function and marks asymptotes, intercepts, and holes.
    """
    try:
        # Validate if the expression has balanced parentheses
        if not is_parentheses_balanced(function_str):
            raise ValueError("Unbalanced parentheses detected in the input.")

        # Enable implicit multiplication
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(function_str, transformations=transformations)

        # Create a numerical function from the symbolic expression
        f = sympy.lambdify(x, expr, modules=['numpy'])

        # Define the plotting range
        x_vals = np.linspace(-10, 10, 1000)
        y_vals = f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='Rational Function', color='blue')

        # Plot vertical asymptotes, horizontal asymptotes, etc.
        # Rest of the plotting logic...

        plt.show()
    except SympifyError:
        raise ValueError("Invalid mathematical expression.")
    except Exception as e:
        raise ValueError(f"Error in plotting rational function: {str(e)}")

# ------------------- Solver Functions ------------------- #
def substitution_method(eq1_str, eq2_str):
    """Solves a system of equations using substitution."""
    A1, B1, C1 = extract_coefficients(eq1_str)
    A2, B2, C2 = extract_coefficients(eq2_str)

    if B1 == 0:
        if A1 == 0:
            raise ValueError("Invalid first equation.")
        # Solve first equation for x
        x = C1 / A1
        # Substitute x into second equation
        y = (C2 - A2 * x) / B2 if B2 != 0 else None
    else:
        # Solve first equation for y
        y = (C1 - A1 * x) / B1
        # Substitute into second equation
        eq2_substituted = simplify(A2 * x + B2 * y - C2)
        # Solve for x
        solution_x = solve(Eq(A2 * x + B2 * ((C1 - A1 * x) / B1), C2), x)
        if not solution_x:
            raise ValueError("No solution found.")
        x_val = float(solution_x[0])
        y_val = (C1 - A1 * x_val) / B1
        return round(x_val, 4), round(y_val, 4)

    return round(x, 4), round(y, 4)

def addition_method(eq1_str, eq2_str):
    """Solves a system of equations using the addition (elimination) method."""
    A1, B1, C1 = extract_coefficients(eq1_str)
    A2, B2, C2 = extract_coefficients(eq2_str)

    # Multiply equations to make coefficients of x opposites
    multiplier = A2 / A1 if A1 != 0 else 0
    if A1 == 0 and A2 == 0:
        # Try eliminating y
        if B1 == 0 and B2 == 0:
            raise ValueError("Cannot eliminate variables; no unique solution.")
        multiplier = B2 / B1 if B1 != 0 else 0
        scaled_A1 = A1 * multiplier
        scaled_B1 = B1 * multiplier
        scaled_C1 = C1 * multiplier
        scaled_A2 = A2
        scaled_B2 = B2
        scaled_C2 = C2
    else:
        scaled_A1 = A1 * multiplier
        scaled_B1 = B1 * multiplier
        scaled_C1 = C1 * multiplier
        scaled_A2 = A2
        scaled_B2 = B2
        scaled_C2 = C2

    # Add the two equations to eliminate x
    delta_A = scaled_A1 + scaled_A2
    delta_B = scaled_B1 + scaled_B2
    delta_C = scaled_C1 + scaled_C2

    if delta_B == 0:
        raise ValueError("No unique solution exists.")

    # Solve for y
    y = delta_C / delta_B

    # Solve for x using the first equation
    if A1 != 0:
        x = (C1 - B1 * y) / A1
    elif A2 != 0:
        x = (C2 - B2 * y) / A2
    else:
        x = None

    return round(x, 4), round(y, 4)

def solve_3x3(eq1_str, eq2_str, eq3_str):
    """Solves a 3x3 system of equations."""
    A1, B1, C1, D1 = extract_coefficients_3x3(eq1_str)
    A2, B2, C2, D2 = extract_coefficients_3x3(eq2_str)
    A3, B3, C3, D3 = extract_coefficients_3x3(eq3_str)

    # Create the coefficient matrix and constant vector
    coeff_matrix = np.array([
        [A1, B1, C1],
        [A2, B2, C2],
        [A3, B3, C3]
    ])
    constants = np.array([D1, D2, D3])

    try:
        solution = np.linalg.solve(coeff_matrix, constants)
        x_val, y_val, z_val = solution
        return round(x_val, 4), round(y_val, 4), round(z_val, 4)
    except np.linalg.LinAlgError:
        raise ValueError("No unique solution exists for the 3x3 system.")

def solve_how_many(price_adult, price_senior, total_people, total_receipts):
    """Solves the 'How Many' problem."""
    # Let A be number of adults and S be number of seniors
    # A + S = total_people
    # price_adult * A + price_senior * S = total_receipts
    A, S = symbols('A S')

    eq1 = Eq(A + S, total_people)
    eq2 = Eq(price_adult * A + price_senior * S, total_receipts)

    solution = solve((eq1, eq2), (A, S))

    if not solution:
        raise ValueError("No solution exists for the 'How Many' problem.")

    adults = solution[A]
    seniors = solution[S]

    if not (is_whole_number(adults) and is_whole_number(seniors)):
        raise ValueError("The solution does not consist of whole numbers.")

    return int(round(adults)), int(round(seniors))

def solve_absolute_value_equation(lhs_abs, lhs_non_abs, operator, rhs):
    """Solves absolute value equations or inequalities."""
    try:
        # Construct the full expression
        if lhs_non_abs:
            expression = simplify(lhs_abs) + simplify(lhs_non_abs)
        else:
            expression = simplify(lhs_abs)

        # Replace '^' with '**' and insert '*' where necessary
        expression = add_multiplication_sign(str(expression))
        rhs = add_multiplication_sign(str(rhs))

        expr_sympy = simplify(expression)

        if operator == "=":
            solutions = solve(Eq(Abs(expr_sympy), float(rhs)), x)
        elif operator == ">":
            # Solve |expr| > rhs => expr > rhs or expr < -rhs
            solutions = solve(Eq(Abs(expr_sympy), float(rhs)), x)
            # Not directly handled; needs to be split into two cases
            solutions = solve(Abs(expr_sympy) > float(rhs), x)
        elif operator == "<":
            # Solve |expr| < rhs => -rhs < expr < rhs
            solutions = solve(Eq(Abs(expr_sympy), float(rhs)), x)
            solutions = solve(Abs(expr_sympy) < float(rhs), x)
        elif operator == ">=":
            # Solve |expr| >= rhs => expr >= rhs or expr <= -rhs
            solutions = solve(Eq(Abs(expr_sympy), float(rhs)), x)
            solutions = solve(Abs(expr_sympy) >= float(rhs), x)
        elif operator == "<=":
            # Solve |expr| <= rhs => -rhs <= expr <= rhs
            solutions = solve(Eq(Abs(expr_sympy), float(rhs)), x)
            solutions = solve(Abs(expr_sympy) <= float(rhs), x)
        else:
            raise ValueError("Invalid operator.")

        return solutions
    except NotImplementedError:
        raise NotImplementedError("Inequalities are not implemented in this solver.")
    except Exception as e:
        raise ValueError(f"Error solving absolute value equation: {str(e)}")

def domain_and_range(points_str):
    """Finds the domain and range from a set of points and checks if it's a function."""
    try:
        # Parse the points string
        points = re.findall(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", points_str)
        if not points:
            raise ValueError("Invalid points format.")

        x_vals = [float(pt[0]) for pt in points]
        y_vals = [float(pt[1]) for pt in points]

        # Determine if it's a function (unique y for each x)
        is_function = len(set(x_vals)) == len(x_vals)

        domain = sorted(set(x_vals))
        range_ = sorted(set(y_vals))

        domain_str = ", ".join(map(str, domain))
        range_str = ", ".join(map(str, range_))

        return domain_str, range_str, is_function
    except Exception as e:
        raise ValueError(f"Error determining domain and range: {str(e)}")

def degrees_radians_conversion(angle_value, input_unit):
    """Converts between degrees and radians."""
    try:
        from sympy import symbols, pi, Rational, simplify
        angle_value = angle_value.strip()

        # Convert the input to a sympy expression
        angle_sympy = simplify(angle_value.replace("^", "**"))

        if input_unit == "Degrees":
            # Convert degrees to radians
            radians = angle_sympy * pi / 180
            radians_simplified = simplify(radians)
            return radians_simplified
        elif input_unit == "Radians":
            # Convert radians to degrees
            degrees = angle_sympy * 180 / pi
            degrees_simplified = simplify(degrees)
            return degrees_simplified
        else:
            raise ValueError("Invalid input unit.")
    except Exception as e:
        raise ValueError(f"Error in degrees-radians conversion: {str(e)}")

def radians_to_dms(angle_in_radians):
    """Converts radians to decimal degrees and then to degrees, minutes, seconds."""
    try:
        from math import degrees
        # Convert radians to decimal degrees
        decimal_degrees = float(angle_in_radians) * 180 / np.pi
        decimal_degrees_rounded = round(decimal_degrees, 4)

        # Extract degrees
        degrees_part = int(decimal_degrees)
        # Calculate fractional part
        fractional_part = decimal_degrees - degrees_part
        # Calculate minutes
        minutes = int(fractional_part * 60)
        # Calculate seconds
        seconds = (fractional_part * 60 - minutes) * 60
        # Round seconds to nearest whole number
        seconds = int(round(seconds))

        # Adjust for rounding issues
        if seconds == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            degrees_part += 1

        return decimal_degrees_rounded, degrees_part, minutes, seconds
    except Exception as e:
        raise ValueError(f"Error in radians to DMS conversion: {str(e)}")

def calculate_arc_length(radius, angle_in_degrees):
    """Calculates the arc length of a circle given radius and central angle in degrees."""
    try:
        from sympy import pi, simplify, S
        # Convert inputs to sympy expressions
        radius = S(radius)  # S() converts string input to sympy number or expression
        angle_in_degrees = S(angle_in_degrees)
        # Convert angle to radians
        angle_in_radians = angle_in_degrees * pi / 180
        angle_in_radians_simplified = simplify(angle_in_radians)
        # Compute the exact arc length
        exact_arc_length = radius * angle_in_radians_simplified
        exact_arc_length_simplified = simplify(exact_arc_length)
        # Compute the approximate arc length
        approximate_arc_length = float(exact_arc_length_simplified.evalf())
        approximate_arc_length_rounded = round(approximate_arc_length, 3)
        return exact_arc_length_simplified, approximate_arc_length_rounded
    except Exception as e:
        raise ValueError(f"Error in arc length calculation: {str(e)}")

def calculate_velocity(radius, rev_per_min):
    """Calculates angular velocity (rad/sec) and linear velocity (units/sec)."""
    try:
        from sympy import pi, S
        # Convert inputs to sympy numbers
        radius = S(radius)
        rev_per_min = S(rev_per_min)
        # Convert rev/min to radians per second
        angular_velocity = (rev_per_min * 2 * pi) / 60  # (rev/min * 2π rad/rev) / 60 sec/min
        angular_velocity_rounded = angular_velocity.evalf(4)  # Round to 4 decimal places
        # Calculate linear velocity
        linear_velocity = angular_velocity * radius  # v = ω * r
        linear_velocity_rounded = linear_velocity.evalf(4)  # Round to 4 decimal places
        return angular_velocity_rounded, linear_velocity_rounded
    except Exception as e:
        raise ValueError(f"Error in velocity calculation: {str(e)}")

def calculate_trig_values(quadrant):
    """Calculates exact trig values for standard angles in the given quadrant."""
    try:
        from sympy import sin, cos, tan, csc, sec, cot, pi, simplify, S

        # Define standard angles in radians
        angles = [0, pi/6, pi/4, pi/3, pi/2]

        # Initialize a dictionary to store results
        trig_values = []

        # Determine sign adjustments based on the quadrant
        # Quadrant I: All positive
        # Quadrant II: sin and csc positive
        # Quadrant III: tan and cot positive
        # Quadrant IV: cos and sec positive

        sign_adjustments = {
            1: {'sin': 1, 'cos': 1, 'tan': 1, 'csc': 1, 'sec': 1, 'cot': 1},
            2: {'sin': 1, 'cos': -1, 'tan': -1, 'csc': 1, 'sec': -1, 'cot': -1},
            3: {'sin': -1, 'cos': -1, 'tan': 1, 'csc': -1, 'sec': -1, 'cot': 1},
            4: {'sin': -1, 'cos': 1, 'tan': -1, 'csc': -1, 'sec': 1, 'cot': -1},
        }

        signs = sign_adjustments.get(quadrant)
        if not signs:
            raise ValueError("Quadrant must be an integer from 1 to 4.")

        for angle in angles:
            # Compute reference angle (always positive in first quadrant)
            ref_angle = angle

            # Calculate exact trig values
            sin_val = simplify(signs['sin'] * sin(ref_angle))
            cos_val = simplify(signs['cos'] * cos(ref_angle))
            tan_val = simplify(signs['tan'] * tan(ref_angle)) if angle != pi/2 else 'Undefined'
            csc_val = simplify(signs['csc'] * csc(ref_angle)) if angle != 0 else 'Undefined'
            sec_val = simplify(signs['sec'] * sec(ref_angle)) if angle != pi/2 else 'Undefined'
            cot_val = simplify(signs['cot'] * cot(ref_angle)) if angle != 0 else 'Undefined'

            # Handle undefined cases
            if angle == 0:
                tan_val = '0'
                cot_val = 'Undefined'
                csc_val = 'Undefined'
            if angle == pi/2:
                tan_val = 'Undefined'
                sec_val = 'Undefined'
                cot_val = '0'

            # Append to the list
            trig_values.append({
                'theta': angle,
                'sin': sin_val,
                'cos': cos_val,
                'tan': tan_val,
                'csc': csc_val,
                'sec': sec_val,
                'cot': cot_val
            })

        return trig_values
    except Exception as e:
        raise ValueError(f"Error in calculating trig values: {str(e)}")

def calculate_trig_functions(angle_str, unit):
    """Calculates the six trigonometric functions for a given angle."""
    try:
        from sympy import sin, cos, tan, csc, sec, cot, pi, simplify, S

        # Convert the angle string to a sympy expression
        angle = S(angle_str)

        # If the angle is in degrees, convert it to radians
        if unit == "Degrees":
            angle = angle * pi / 180

        # Simplify the angle
        angle = simplify(angle)

        # Calculate exact trig values
        sin_val = simplify(sin(angle))
        cos_val = simplify(cos(angle))
        tan_val = simplify(tan(angle)) if cos(angle) != 0 else 'Undefined'
        csc_val = simplify(csc(angle)) if sin(angle) != 0 else 'Undefined'
        sec_val = simplify(sec(angle)) if cos(angle) != 0 else 'Undefined'
        cot_val = simplify(cot(angle)) if sin(angle) != 0 else 'Undefined'

        return {
            'sin': sin_val,
            'cos': cos_val,
            'tan': tan_val,
            'csc': csc_val,
            'sec': sec_val,
            'cot': cot_val
        }
    except Exception as e:
        raise ValueError(f"Error in calculating trigonometric functions: {str(e)}")

def calculate_trig_ratios_xy(x_value, y_value):
    """Calculates trigonometric ratios given x and y coordinates with exact simplified answers."""
    try:
        from sympy import sqrt, simplify, S, Rational
        x = S(x_value)
        y = S(y_value)
        r = simplify(sqrt(x**2 + y**2))
        if r == 0:
            raise ValueError("r cannot be zero.")
        sin_theta = simplify(y / r)
        cos_theta = simplify(x / r)
        tan_theta = simplify(y / x) if x != 0 else 'Undefined'
        csc_theta = simplify(r / y) if y != 0 else 'Undefined'
        sec_theta = simplify(r / x) if x != 0 else 'Undefined'
        cot_theta = simplify(x / y) if y != 0 else 'Undefined'
        return {
            'sin': sin_theta,
            'cos': cos_theta,
            'tan': tan_theta,
            'csc': csc_theta,
            'sec': sec_theta,
            'cot': cot_theta,
            'r': r
        }
    except Exception as e:
        raise ValueError(f"Error in calculating trigonometric ratios: {str(e)}")

def calculate_remaining_trig_functions(known_func_name, known_value_str, quadrant_str):
    """Calculates the remaining trig functions given one trig function value and the quadrant."""
    from sympy import sin, cos, tan, csc, sec, cot, sqrt, simplify, S
    known_value = S(known_value_str)  # Convert string to sympy number
    quadrant = int(quadrant_str)

    # Map the function names to sympy functions
    trig_functions = {
        'sin': sin,
        'cos': cos,
        'tan': tan,
        'csc': csc,
        'sec': sec,
        'cot': cot
    }

    # We need to find sin θ and cos θ first
    if known_func_name == 'sin':
        sin_theta = known_value
        # Use Pythagorean identity: sin^2 θ + cos^2 θ = 1
        cos_theta_sq = 1 - sin_theta**2
        cos_theta = sqrt(cos_theta_sq)
    elif known_func_name == 'cos':
        cos_theta = known_value
        sin_theta_sq = 1 - cos_theta**2
        sin_theta = sqrt(sin_theta_sq)
    elif known_func_name == 'tan':
        tan_theta = known_value
        # tan θ = sin θ / cos θ
        # Use identity 1 + tan^2 θ = sec^2 θ
        sec_theta_sq = 1 + tan_theta**2
        sec_theta = sqrt(sec_theta_sq)
        cos_theta = 1 / sec_theta
        sin_theta = tan_theta * cos_theta
    elif known_func_name == 'csc':
        sin_theta = 1 / known_value
        cos_theta_sq = 1 - sin_theta**2
        cos_theta = sqrt(cos_theta_sq)
    elif known_func_name == 'sec':
        cos_theta = 1 / known_value
        sin_theta_sq = 1 - cos_theta**2
        sin_theta = sqrt(sin_theta_sq)
    elif known_func_name == 'cot':
        cot_theta = known_value
        # cot θ = cos θ / sin θ
        # Use identity 1 + cot^2 θ = csc^2 θ
        csc_theta_sq = 1 + cot_theta**2
        csc_theta = sqrt(csc_theta_sq)
        sin_theta = 1 / csc_theta
        cos_theta = cot_theta * sin_theta
    else:
        raise ValueError("Invalid trigonometric function name.")

    # Determine the signs based on the quadrant
    # Quadrant I: sin > 0, cos > 0
    # Quadrant II: sin > 0, cos < 0
    # Quadrant III: sin < 0, cos < 0
    # Quadrant IV: sin < 0, cos > 0
    if quadrant == 1:
        sin_theta = simplify(sin_theta)
        cos_theta = simplify(cos_theta)
    elif quadrant == 2:
        sin_theta = simplify(sin_theta)
        cos_theta = simplify(-cos_theta)
    elif quadrant == 3:
        sin_theta = simplify(-sin_theta)
        cos_theta = simplify(-cos_theta)
    elif quadrant == 4:
        sin_theta = simplify(-sin_theta)
        cos_theta = simplify(cos_theta)
    else:
        raise ValueError("Quadrant must be an integer from 1 to 4.")

    # Now compute the other functions
    tan_theta = simplify(sin_theta / cos_theta) if cos_theta != 0 else 'Undefined'
    csc_theta = simplify(1 / sin_theta) if sin_theta != 0 else 'Undefined'
    sec_theta = simplify(1 / cos_theta) if cos_theta != 0 else 'Undefined'
    cot_theta = simplify(cos_theta / sin_theta) if sin_theta != 0 else 'Undefined'

    # Build the results dictionary
    results = {
        'sin': sin_theta,
        'cos': cos_theta,
        'tan': tan_theta,
        'csc': csc_theta,
        'sec': sec_theta,
        'cot': cot_theta
    }

    return results

def calculate_amplitude_and_period(equation_str):
    """Calculates period, phase shift, vertical shift, and key points of a trigonometric function and plots it."""
    try:
        from sympy import symbols, pi, sin, cos, tan, cot, sec, csc, Abs, S, expand, lambdify
        import numpy as np
        import matplotlib.pyplot as plt
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application
        )

        # Remove spaces and replace '^' with '**' for exponentiation
        equation_str = equation_str.replace(" ", "").replace("^", "**")

        # Ensure the equation is in the form y = ...
        if '=' not in equation_str:
            raise ValueError("Equation must contain '='.")
        lhs, rhs = equation_str.split('=')
        if lhs != 'y':
            raise ValueError("Equation must be in the form y = ...")

        # Define the variable x and ensure it's real
        x = symbols('x', real=True)

        # Parse RHS expression using sympy with implicit multiplication and disable evaluation
        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {
            'x': x, 'pi': pi, 'sin': sin, 'cos': cos, 'tan': tan,
            'cot': cot, 'sec': sec, 'csc': csc
        }

        # Parse the RHS without automatic simplification
        rhs_expr = parse_expr(rhs, local_dict=local_dict, transformations=transformations, evaluate=False)

        # Check for vertical shift
        if rhs_expr.is_Add:
            terms = rhs_expr.as_ordered_terms()
            trig_term = None
            vertical_shift = S.Zero
            for term in terms:
                if any(term.has(func) for func in [sin, cos, tan, cot, sec, csc]):
                    trig_term = term
                else:
                    vertical_shift += term
        else:
            trig_term = rhs_expr
            vertical_shift = S.Zero

        # Extract amplitude or leading coefficient (skipped for tan)
        amplitude = S.One
        if trig_term.is_Mul:
            factors = trig_term.args
            for factor in factors:
                if any(factor.has(func) for func in [sin, cos, tan, cot, sec, csc]):
                    trig_function_term = factor
                else:
                    amplitude *= factor
        else:
            trig_function_term = trig_term

        amplitude_value = Abs(amplitude)

        # Identify the trigonometric function
        trig_function = trig_function_term.func

        # Extract argument of the trigonometric function (it will be of the form B*x + D)
        argument = trig_function_term.args[0]
        argument = expand(argument)

        # Extract B (coefficient of x) and D (constant term)
        B = argument.coeff(x)  # Coefficient of x is the frequency B
        D = argument - B * x  # Constant term is D
        phase_shift_value = -D / B  # Phase shift C = -D/B

        # Calculate the period
        if trig_function in [sin, cos, sec, csc]:
            base_period = 2 * pi
        elif trig_function in [tan, cot]:
            base_period = pi
        else:
            raise ValueError("Unsupported trigonometric function.")

        period = base_period / Abs(B)

        # Determine phase shift direction
        phase_shift_num = float(phase_shift_value.evalf())
        if phase_shift_num > 0:
            phase_shift_direction = f"right by {Abs(phase_shift_value)}"
        elif phase_shift_num < 0:
            phase_shift_direction = f"left by {Abs(phase_shift_value)}"
        else:
            phase_shift_direction = "no horizontal shift"

        # Determine vertical shift direction
        vertical_shift_value = float(vertical_shift.evalf())
        if vertical_shift_value > 0:
            vertical_shift_direction = f"up by {vertical_shift}"
        elif vertical_shift_value < 0:
            vertical_shift_direction = f"down by {Abs(vertical_shift)}"
        else:
            vertical_shift_direction = "no vertical shift"

        # Initialize key points list
        key_points = []

        if trig_function == tan:
            # Handle tangent function key points
            left_asymptote_x = -pi / (2 * B) - D / B
            right_asymptote_x = pi / (2 * B) - D / B

            center_x = 0
            center_value = vertical_shift  # tan(0) = 0

            left_mid_x = left_asymptote_x / 2
            right_mid_x = right_asymptote_x / 2

            left_mid_value = amplitude * tan(B * left_mid_x + D) + vertical_shift
            right_mid_value = amplitude * tan(B * right_mid_x + D) + vertical_shift

            key_points = [
                (left_asymptote_x, S.NaN),    # Left asymptote
                (left_mid_x, left_mid_value),  # Left midpoint
                (center_x, center_value),      # Center point
                (right_mid_x, right_mid_value),  # Right midpoint
                (right_asymptote_x, S.NaN)     # Right asymptote
            ]
        elif trig_function in [sin, cos]:
            # Handle sine and cosine function key points
            # One period key points
            h = phase_shift_value
            # Critical points within one period
            key_points = [
                (h - period/4, vertical_shift),
                (h, amplitude + vertical_shift),
                (h + period/4, vertical_shift),
                (h + period/2, -amplitude + vertical_shift),
                (h + 3*period/4, vertical_shift),
                (h + period, amplitude + vertical_shift)
            ]
        # You can add more conditions for other trig functions if needed

        # Define custom numpy functions
        import numpy as np

        def numpy_cot(x):
            return 1 / np.tan(x)

        def numpy_sec(x):
            return 1 / np.cos(x)

        def numpy_csc(x):
            return 1 / np.sin(x)

        # Create a dictionary for lambdify
        numpy_funcs = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'cot': numpy_cot,
            'sec': numpy_sec,
            'csc': numpy_csc,
            'pi': np.pi,
        }

        # Create a numerical function using lambdify
        f = lambdify(x, rhs_expr, modules=[numpy_funcs])

        # Generate x-values for plotting
        if trig_function == tan:
            x_min = float(left_asymptote_x - period)
            x_max = float(right_asymptote_x + period)
        else:
            x_min = float(phase_shift_value - 2 * period)
            x_max = float(phase_shift_value + 2 * period)
        x_values = np.linspace(x_min, x_max, 1000)

        # Evaluate the function numerically
        y_values = f(x_values)

        # Handle complex numbers in y_values
        y_real = np.real(y_values)
        y_imag = np.imag(y_values)
        imag_threshold = 1e-6
        mask = np.abs(y_imag) > imag_threshold
        y_real[mask] = np.nan
        y_values = y_real

        # Handle asymptotes by setting large values to NaN
        y_values[np.abs(y_values) > 10 * abs(float(amplitude_value)) + abs(float(vertical_shift_value))] = np.nan

        # Plot the function
        plt.figure(figsize=(8, 4))
        plt.plot(x_values, y_values, label=equation_str)
        plt.title(f"Graph of {equation_str}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # Set y-limits based on amplitude and vertical shift
        if trig_function in [sin, cos]:
            y_min = -1.5 * float(amplitude_value) + float(vertical_shift_value)
            y_max = 1.5 * float(amplitude_value) + float(vertical_shift_value)
        elif trig_function == tan:
            y_min = -10 * float(amplitude_value) + float(vertical_shift_value)
            y_max = 10 * float(amplitude_value) + float(vertical_shift_value)
        else:
            y_min = -10
            y_max = 10
        plt.ylim(y_min, y_max)

        # Plot key points and asymptotes
        for x_kp, y_kp in key_points:
            x_val = float(x_kp.evalf()) if hasattr(x_kp, 'evalf') else float(x_kp)
            y_kp_eval = y_kp.evalf() if hasattr(y_kp, 'evalf') else y_kp
            if isinstance(y_kp_eval, sympy.Expr) and y_kp_eval.is_real:
                y_val = float(y_kp_eval)
            elif y_kp_eval == S.NaN:
                y_val = np.nan
            else:
                y_val = np.nan
            if np.isnan(y_val):
                plt.axvline(x=x_val, color='gray', linestyle='--')  # Plot asymptotes
            else:
                plt.plot(x_val, y_val, 'ro')
                plt.annotate(f"({x_val:.2f}, {y_val:.2f})", xy=(x_val, y_val),
                             textcoords="offset points", xytext=(0, 10), ha='center')

        plt.legend()
        plt.show()

        # Return the amplitude (or None for tangent), period, phase shift with direction, vertical shift, and key points
        phase_shift_description = f"{phase_shift_direction}, {vertical_shift_direction}"
        return amplitude_value if trig_function != tan else None, period, phase_shift_description, key_points

    except Exception as e:
        raise ValueError(f"Error in calculating amplitude and period: {str(e)}")

def evaluate_piecewise_function_defined(expr1, op1, val1, expr2, op2, val2):
    """
    Evaluates and plots the piecewise function when it is defined outside the given intervals.
    """
    try:
        # Importing necessary modules
        from sympy import symbols, sympify, Piecewise, lambdify, Eq
        import numpy as np
        import matplotlib.pyplot as plt

        # Convert the input expressions to sympy expressions
        f_expr1 = validate_expression(expr1)
        f_expr2 = validate_expression(expr2)

        # Convert the condition values to sympy numbers
        val1 = validate_numeric(val1)
        val2 = validate_numeric(val2)

        # Create conditions using sympy relational operators
        cond1 = eval(f"x {validate_operator(op1)} {val1}")
        cond2 = eval(f"x {validate_operator(op2)} {val2}")

        # Define the piecewise function
        f_piecewise = Piecewise(
            (f_expr1, cond1),
            (f_expr2, cond2),
            (0, True)  # Else, defined as 0
        )

        # Prepare for plotting
        f_lambdified = lambdify(x, f_piecewise, modules=['numpy'])
        x_min = min(val1, val2) - 10
        x_max = max(val1, val2) + 10
        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = f_lambdified(x_vals)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='Piecewise Function', color='blue')
        plt.title('Graph of the Piecewise Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

        return "The piecewise function has been graphed successfully."

    except Exception as e:
        raise ValueError(f"Error in evaluating piecewise function: {str(e)}")

def evaluate_piecewise_function_undefined(expr1, expr2, expr3, x_val):
    """
    Evaluates the piecewise function at a given x value when undefined outside intervals.
    """
    try:
        # Convert the input expressions to sympy expressions
        f_expr1 = validate_expression(expr1)
        f_expr2 = validate_expression(expr2)
        f_expr3 = validate_expression(expr3)

        # Define conditions
        cond1 = x < 0
        cond2 = Eq(x, 0)
        cond3 = x > 0

        # Define the piecewise function
        f_piecewise = Piecewise(
            (f_expr1, cond1),
            (f_expr2, cond2),
            (f_expr3, cond3),
            (sympy.nan, True)  # Else undefined
        )

        # Evaluate the function at x_val
        result = f_piecewise.subs(x, x_val).evalf()

        return result

    except Exception as e:
        raise ValueError(f"Error in evaluating piecewise function: {str(e)}")

def solve_equation(lhs_str, rhs_str):
    """Solves the equation lhs = rhs for x, including equations with square roots."""
    try:
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        # Prepare transformations and local dictionary
        transformations = standard_transformations + (implicit_multiplication_application,)
        local_dict = {'x': x, 'sqrt': sqrt, 'sin': sin, 'cos': cos, 'tan': tan, 'csc': csc, 'sec': sec, 'cot': cot, 'pi': pi}
        # Convert the input strings to sympy expressions
        lhs_expr = parse_expr(add_multiplication_sign(lhs_str), local_dict=local_dict, transformations=transformations)
        rhs_expr = parse_expr(add_multiplication_sign(rhs_str), local_dict=local_dict, transformations=transformations)
        # Solve the equation lhs = rhs
        solutions = solve(Eq(lhs_expr, rhs_expr), x)
        # Filter out extraneous solutions if necessary
        valid_solutions = []
        for sol in solutions:
            # Check if the solution satisfies the original equation
            if lhs_expr.subs(x, sol).equals(rhs_expr.subs(x, sol)):
                valid_solutions.append(sol)
        if not valid_solutions:
            return "No solution"
        else:
            return valid_solutions
    except Exception as e:
        raise ValueError(f"Error solving equation: {str(e)}")

def solve_quadratics(equation_str):
    """
    Analyzes a quadratic equation and returns its key properties.
    """
    try:
        # Remove 'y=' or 'y =' from the equation
        equation_str = equation_str.lower().replace('y=', '').replace('y =', '').strip()

        # Convert to SymPy expression
        expr = validate_expression(equation_str)

        # Ensure the expression is quadratic
        poly = Poly(expr, x)
        if poly.degree() != 2:
            raise ValueError("The equation is not a quadratic function.")

        # Extract coefficients
        a, b, c = poly.all_coeffs()

        # Vertex (h, k)
        h = -b / (2 * a)
        k = expr.subs(x, h).evalf()

        # Axis of symmetry
        axis_of_symmetry = h

        # Concavity
        concavity = "Upward" if a > 0 else "Downward"

        # Discriminant to find x-intercepts
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            x1 = (-b + sympy.sqrt(discriminant)) / (2*a)
            x2 = (-b - sympy.sqrt(discriminant)) / (2*a)
            x_intercepts = [x1.evalf(), x2.evalf()]
        elif discriminant == 0:
            x_root = -b / (2*a)
            x_intercepts = [x_root.evalf()]
        else:
            x_intercepts = []  # No real roots

        # Y-intercept
        y_intercept = c

        # Domain
        domain = "All real numbers"

        # Range
        if concavity == "Upward":
            range_ = f"[{format_solution(k)}, ∞)"
        else:
            range_ = f"(-∞, {format_solution(k)}]"

        # Intervals of increase and decrease
        if concavity == "Upward":
            decreasing_intervals = f"(-∞, {format_solution(h)})"
            increasing_intervals = f"({format_solution(h)}, ∞)"
        else:
            increasing_intervals = f"(-∞, {format_solution(h)})"
            decreasing_intervals = f"({format_solution(h)}, ∞)"

        return {
            "vertex": (format_solution(h), format_solution(k)),
            "axis_of_symmetry": format_solution(axis_of_symmetry),
            "concavity": concavity,
            "x_intercepts": [format_solution(sol) for sol in x_intercepts],
            "y_intercept": format_solution(y_intercept),
            "domain": domain,
            "range": range_,
            "increasing_intervals": increasing_intervals,
            "decreasing_intervals": decreasing_intervals
        }
    except Exception as e:
        raise ValueError(f"Error in solving quadratics: {str(e)}")

def solve_polynomial(polynomial_str):
    """Solves f(x) = 0, factors the polynomial, and plots it."""
    try:
        # Convert the input string to a SymPy expression
        polynomial_expr = sympify(add_multiplication_sign(polynomial_str))

        # Factor the polynomial
        factored_form = factor(polynomial_expr)

        # Solve f(x) = 0 using the global 'x'
        solutions = solve(polynomial_expr, x)

        # Plot the polynomial
        plot_polynomial(polynomial_expr, solutions)

        return factored_form, solutions
    except Exception as e:
        raise ValueError(f"Error in solving polynomial: {str(e)}")

def neg_arc_function(function_name, mode, value_str):
    """
    Computes the inverse trigonometric function value.

    Parameters:
    - function_name: 'sin', 'cos', 'tan', 'csc', 'sec', 'cot'
    - mode: '-1' or 'arc'
    - value_str: The value as a string (e.g., '1', 'sqrt(3)/2')

    Returns:
    - A tuple (angle_rad, angle_deg) where:
      - angle_rad: The angle in radians (simplified if possible), or numeric value
      - angle_deg: The angle in degrees (evaluated numerically)
    - Returns 'No solution' if the input is outside the domain.
    """
    try:
        from sympy import S, pi, asin, acos, atan, acsc, asec, acot, sqrt, N
        value = S(value_str)
        # Map function names to inverse trigonometric functions
        inverse_functions = {
            'sin': asin,
            'cos': acos,
            'tan': atan,
            'csc': acsc,
            'sec': asec,
            'cot': acot
        }
        if function_name not in inverse_functions:
            raise ValueError("Invalid function name.")
        inverse_func = inverse_functions[function_name]
        # Compute the inverse trigonometric function
        angle_rad = inverse_func(value)
        if angle_rad.has('I'):
            # The result is complex, so input is outside the domain
            return "No solution"
        # Try to simplify the radian value
        angle_rad_simplified = angle_rad.rewrite(pi).simplify()
        # Check if angle_rad_simplified contains inverse trig functions
        if angle_rad_simplified.has(asin, acos, atan, acsc, asec, acot):
            # Cannot simplify further, evaluate numerically
            angle_rad_final = angle_rad.evalf(5)  # Evaluate to 5 decimal places
        else:
            angle_rad_final = angle_rad_simplified
        # Convert angle to degrees
        angle_deg = angle_rad.evalf(5) * 180 / pi
        angle_deg = angle_deg.evalf(5)
        return angle_rad_final, angle_deg
    except Exception as e:
        return "No solution"

def right_triangle(given_values):
    """
    Given any two of the sides (a, b, c) or angles (A, B), computes the remaining sides and angles.

    Parameters:
    - given_values: dictionary with two keys and their values, e.g., {'a': 3, 'b': 4}

    Returns:
    - Dictionary with all sides and angles
    """
    import math

    # Initialize variables
    a = b = c = A = B = None

    # Extract given values
    for key, value in given_values.items():
        if key == 'a':
            a = float(value)
        elif key == 'b':
            b = float(value)
        elif key == 'c':
            c = float(value)
        elif key == 'A':
            A = float(value)
        elif key == 'B':
            B = float(value)
        else:
            raise ValueError("Invalid key in given_values")

    # Since C = 90 degrees
    C = 90.0

    # Compute missing angles if one is given
    if A is not None and B is None:
        B = 90.0 - A
    elif B is not None and A is None:
        A = 90.0 - B
    elif A is not None and B is not None:
        # Validate that A + B = 90 degrees
        if abs(A + B - 90.0) > 1e-6:
            raise ValueError("Angles A and B must add up to 90 degrees in a right triangle")

    # Compute sides and angles based on given information
    if a is not None and b is not None:
        # Given sides a and b
        c = math.hypot(a, b)
        A = math.degrees(math.atan2(a, b))
        B = 90.0 - A
    elif a is not None and c is not None:
        # Given sides a and c
        if c <= a:
            raise ValueError("Hypotenuse c must be greater than side a")
        b = math.sqrt(c**2 - a**2)
        A = math.degrees(math.asin(a / c))
        B = 90.0 - A
    elif b is not None and c is not None:
        # Given sides b and c
        if c <= b:
            raise ValueError("Hypotenuse c must be greater than side b")
        a = math.sqrt(c**2 - b**2)
        B = math.degrees(math.asin(b / c))
        A = 90.0 - B
    elif a is not None and A is not None:
        # Given side a and angle A
        c = a / math.sin(math.radians(A))
        b = math.sqrt(c**2 - a**2)
        B = 90.0 - A
    elif b is not None and B is not None:
        # Given side b and angle B
        c = b / math.sin(math.radians(B))
        a = math.sqrt(c**2 - b**2)
        A = 90.0 - B
    elif a is not None and B is not None:
        # Given side a and angle B
        c = a / math.cos(math.radians(B))
        b = math.sqrt(c**2 - a**2)
        A = 90.0 - B
    elif b is not None and A is not None:
        # Given side b and angle A
        c = b / math.cos(math.radians(A))
        a = math.sqrt(c**2 - b**2)
        B = 90.0 - A
    elif c is not None and A is not None:
        # Given hypotenuse c and angle A
        a = c * math.sin(math.radians(A))
        b = c * math.cos(math.radians(A))
        B = 90.0 - A
    elif c is not None and B is not None:
        # Given hypotenuse c and angle B
        b = c * math.sin(math.radians(B))
        a = c * math.cos(math.radians(B))
        A = 90.0 - B
    else:
        raise ValueError("Insufficient or invalid data to solve the triangle")

    # Build result dictionary
    result = {
        'a': a,
        'b': b,
        'c': c,
        'A': A,
        'B': B,
        'C': C
    }
    return result

def solve_trig_intervals(equation_str, interval_unit):
    """
    Solves trigonometric equations with one or more trigonometric functions within a specified interval.

    Parameters:
    - equation_str: str, the trigonometric equation as a string (e.g., "2*sin(x) + 3*cos(x) = 1")
    - interval_unit: str, the unit of the interval ("Radians", "Degrees", "Degrees (0 to 180)")

    Returns:
    - dict containing:
        - 'transformed_eq': str, the standardized form of the equation after rearrangement
        - 'solutions': list of str, solutions within the specified interval, formatted with units
    """
    import sympy
    from sympy import symbols, Eq, pi, sin, cos, tan, csc, sec, cot, solveset, Interval, Function, nsimplify, simplify
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    import re

    # Define the variable
    x = symbols('x', real=True)

    # Replace '^' with '**' for exponentiation
    equation_str = equation_str.replace('^', '**')

    # Handle trigonometric functions with degrees, e.g., sin(48.5') -> sin(48.5 * pi / 180)
    equation_str = re.sub(r'(sin|cos|tan|csc|sec|cot)\(\s*([0-9.]+)\'\s*\)', r'\1(\2 * pi / 180)', equation_str)

    # Split the equation into LHS and RHS
    if '=' not in equation_str:
        raise ValueError("Equation must contain '='.")

    parts = equation_str.split('=')
    if len(parts) != 2:
        raise ValueError("Equation must contain exactly one '=' sign.")
    lhs_str, rhs_str = parts

    # Define parsing transformations
    transformations = standard_transformations + (implicit_multiplication_application,)

    # Define local dictionary for parsing
    local_dict = {
        'x': x,
        'pi': pi,
        'sin': sin,
        'cos': cos,
        'tan': tan,
        'csc': csc,
        'sec': sec,
        'cot': cot,
        'sqrt': sympy.sqrt,
    }

    try:
        # Parse LHS and RHS expressions
        lhs_expr = parse_expr(lhs_str, local_dict=local_dict, transformations=transformations)
        rhs_expr = parse_expr(rhs_str, local_dict=local_dict, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Error parsing the equation: {e}")

    # Create the equation
    eq = Eq(lhs_expr, rhs_expr)

    # Convert the equation to standard form (lhs - rhs = 0)
    eq_standard = Eq(lhs_expr - rhs_expr, 0)

    # Handle degrees or radians
    if interval_unit == 'Degrees':
        x_deg = symbols('x_deg', real=True)
        eq_standard = eq_standard.subs(x, x_deg * pi / 180)
        x_new = x_deg
        domain = Interval(0, 360, False, True)
        unit = "degrees"
    elif interval_unit == 'Degrees (0 to 180)':
        x_deg = symbols('x_deg', real=True)
        eq_standard = eq_standard.subs(x, x_deg * pi / 180)
        x_new = x_deg
        domain = Interval(0, 180, False, True)
        unit = "degrees"
    elif interval_unit == 'Radians':
        # No substitution needed
        x_new = x
        domain = Interval(0, 2 * pi, False, True)
        unit = "radians"
    else:
        raise ValueError("Invalid interval unit. Choose 'Radians', 'Degrees', or 'Degrees (0 to 180)'.")

    # Solve the equation within the domain
    solutions = solveset(eq_standard, x_new, domain=domain)

    # Convert solutions to a list
    if solutions.is_FiniteSet:
        solutions_list = list(solutions)
    else:
        # Handle ImageSet or other types if necessary
        solutions_list = []

    # Remove duplicates without using set()
    unique_solutions = []
    for sol in solutions_list:
        if not any(sol.equals(s) for s in unique_solutions):
            unique_solutions.append(sol)
    solutions_list = unique_solutions

    # Sort the solutions
    try:
        solutions_list.sort(key=lambda s: float(s.evalf()))
    except:
        pass  # In case solutions can't be converted to float for sorting

    # Format solutions with appropriate units and precision
    final_solutions = []
    for sol in solutions_list:
        if unit == "degrees":
            sol_eval = sol.evalf()
            # Round to two decimal places
            sol_rounded = round(float(sol_eval), 2)
            final_solutions.append(f"{sol_rounded} degrees")
        else:
            # Express in terms of pi for radians
            try:
                sol_simplified = nsimplify(sol / pi, tolerance=1e-10) * pi
                final_solutions.append(f"{sol_simplified} radians")
            except:
                # Fallback to numerical value if simplification fails
                sol_num = sol.evalf(5)
                final_solutions.append(f"{sol_num} radians")

    return {
        'transformed_eq': str(eq_standard),
        'solutions': final_solutions
    }

def find_sinusoidal_equation(y_intercept, trough_y, trough_x, first_wave_x):
    """
    Finds the equation of a sine wave given the y-intercept, trough, and first wave end point.
    Returns the equation of the form y = A * sin(Bx + C) + D, supporting fractional input.
    """
    from sympy import symbols, sin, solve, pi, Eq

    # Define the variable
    x = symbols('x')

    # Convert inputs to fractions for precision
    y_intercept = Fraction(y_intercept)
    trough_y = Fraction(trough_y)
    trough_x = Fraction(trough_x)
    first_wave_x = Fraction(first_wave_x)

    # Amplitude A
    A = abs(trough_y - y_intercept) / 2  # Amplitude is half the distance between trough and y-intercept

    # Vertical shift D (midline)
    D = (trough_y + y_intercept) / 2

    # Period and frequency B
    period = 2 * (first_wave_x - trough_x)  # The period is double the distance from trough to first wave peak
    B = 2 * pi / period  # B is calculated using 2π / period

    # Phase shift C (based on the location of the trough)
    C = solve(Eq(A * sin(B * trough_x + symbols('C')), trough_y - D), symbols('C'))[0]

    # Construct the sinusoidal equation
    equation = A * sin(B * x + C) + D

    return equation.simplify()

def divide_polynomials(equation_str, divisor_str):
    """
    Divides a polynomial by a divisor. Performs synthetic division if the divisor is linear.
    Otherwise, performs polynomial long division and returns the quotient and remainder.

    Parameters:
    - equation_str: str, the dividend polynomial (e.g., "10x^4 + 4x^3 - 9x^2 + 3")
    - divisor_str: str, the divisor polynomial (e.g., "x - 3" or "2x^2 -1")

    Returns:
    - tuple: (quotient, remainder)
      - quotient: sympy expression representing the quotient polynomial
      - remainder: sympy expression representing the remainder polynomial
    """
    try:
        # Preprocess inputs: Replace '^' with '**' and remove spaces
        equation_str = equation_str.replace('^', '**').replace(' ', '')
        divisor_str = divisor_str.replace('^', '**').replace(' ', '')

        # Insert '*' where necessary (e.g., '2x' -> '2*x')
        equation_str = add_multiplication_sign(equation_str)
        divisor_str = add_multiplication_sign(divisor_str)

        # Parse the polynomial and divisor expressions using sympy
        transformations = standard_transformations + (implicit_multiplication_application,)
        poly_expr = parse_expr(equation_str, transformations=transformations)
        divisor_expr = parse_expr(divisor_str, transformations=transformations)

        # Convert to Poly objects with respect to x
        poly = Poly(poly_expr, x)
        divisor = Poly(divisor_expr, x)

        # Determine the degree of the divisor
        divisor_degree = divisor.degree()

        if divisor_degree == 1:
            # Synthetic Division

            # Extract coefficients (from highest degree to constant term)
            coefficients = poly.all_coeffs()

            # Parse the divisor to extract c from "x - c" or "x + c"
            match = re.match(r"x([+-])([\d./]+)", divisor_str)
            if not match:
                raise ValueError("For synthetic division, the divisor must be in the form 'x - c' or 'x + c'.")

            sign, num_str = match.groups()
            c = float(num_str) if sign == '-' else -float(num_str)

            # Perform synthetic division
            synthetic = [coefficients[0]]  # Initialize with the leading coefficient
            for coeff in coefficients[1:]:
                synthetic.append(coeff + synthetic[-1] * c)

            remainder = synthetic.pop()  # Last element is the remainder
            quotient_coeffs = synthetic  # Remaining elements are the quotient coefficients

            # Create the quotient polynomial
            quotient = Poly(quotient_coeffs, x).as_expr()

            return quotient, remainder

        else:
            # Polynomial Long Division

            # Perform polynomial long division
            quotient, remainder = poly.div(divisor)

            # Convert quotient and remainder to expressions
            quotient_expr = quotient.as_expr()
            remainder_expr = remainder.as_expr()

            # Always return quotient and remainder
            return quotient_expr, remainder_expr

    except Exception as e:
        raise ValueError(f"Error in dividing polynomials: {str(e)}")

def rational_zeros(polynomial_str):
    """
    Uses the Rational Zeros Theorem to list all potential rational zeros of the given polynomial,
    finds the actual rational zeros, performs polynomial long division to reduce the polynomial,
    finds remaining zeros (which may be irrational), and provides the complete factored form.

    Parameters:
    - polynomial_str: str, the polynomial expression (e.g., "x^3 -7x^2 -4x +28")

    Returns:
    - dict containing:
        - 'potential_zeros': list of potential rational zeros
        - 'actual_rational_zeros': list of actual rational zeros found
        - 'remaining_zeros': list of remaining zeros (irrational or complex)
        - 'factored_form': factored form of the polynomial as a SymPy expression
    """
    try:
        # Preprocess the polynomial string: replace '^' with '**' and insert '*' where necessary
        poly_str = add_multiplication_sign(polynomial_str.replace('^', '**').replace(' ', ''))

        # Parse the polynomial expression using SymPy
        poly_expr = parse_expr(poly_str, transformations=standard_transformations + (implicit_multiplication_application,))
        poly = Poly(poly_expr, x)

        # Extract coefficients: an (leading coefficient) and a0 (constant term)
        coefficients = poly.all_coeffs()
        an = coefficients[0]  # Leading coefficient
        a0 = coefficients[-1]  # Constant term

        # Function to get all integer factors of a number, including both positive and negative
        def get_factors(n):
            return sorted(set(sympy.divisors(abs(n)) + [-d for d in sympy.divisors(abs(n))]))

        # Get all possible p's and q's
        p_factors = get_factors(a0)
        q_factors = get_factors(an)

        # Generate all possible p/q in lowest terms
        potential_zeros = set()
        for p in p_factors:
            for q in q_factors:
                if q != 0:
                    frac = Fraction(p, q).limit_denominator()
                    potential_zeros.add(frac)

        # Sort the potential zeros
        potential_zeros = sorted(potential_zeros, key=lambda f: (f.numerator / f.denominator))

        # Convert fractions to SymPy Rational objects for consistency
        potential_zeros_sympy = [sympy.Rational(f.numerator, f.denominator) for f in potential_zeros]

        # Initialize list for actual rational zeros
        actual_rational_zeros = []

        # Initialize factored expression as 1 (no factors yet)
        factored_expr = 1

        # Initialize a copy of the polynomial for division
        current_poly = poly

        # Iterate through potential zeros and check if they are actual zeros
        for zero in potential_zeros_sympy:
            # Check if zero is a root
            if current_poly.eval(zero) == 0:
                actual_rational_zeros.append(zero)
                # Multiply the factored expression by (x - zero)
                factored_expr *= (x - zero)
                # Divide the current polynomial by (x - zero)
                divisor = Poly(x - zero, x)
                quotient, remainder = current_poly.div(divisor)
                if remainder.as_expr() != 0:
                    # This should not happen if zero is a root
                    raise ValueError(f"Division by (x - {zero}) did not result in zero remainder.")
                # Update current_poly to the quotient for further factoring
                current_poly = quotient

        # After factoring out all rational zeros, solve the reduced polynomial for remaining zeros
        remaining_zeros = []
        if current_poly.degree() > 0:
            # Solve for remaining zeros (real and complex)
            remaining_zeros = list(sympy.solveset(current_poly.as_expr(), x, domain=sympy.S.Complexes))
            # Convert to SymPy Rational or other types if necessary
            # (Already handled by solveset)

        # Construct the complete factored form
        factored_form = factored_expr
        for rem_zero in remaining_zeros:
            factored_form *= (x - rem_zero)

        # Simplify the factored form
        factored_form = sympy.expand(factored_form)

        return {
            'potential_zeros': potential_zeros_sympy,
            'actual_rational_zeros': actual_rational_zeros,
            'remaining_zeros': remaining_zeros,
            'factored_form': factored_form
        }

    except Exception as e:
        raise ValueError(f"Error in finding rational zeros: {str(e)}")

def solve_inequalities(left_expr, operator, right_value):
    """
    Solves the given inequality and returns the solution in interval notation.

    Parameters:
    - left_expr: str, the expression on the left side of the inequality (e.g., "(x-3)*(x-4)*(x-5)")
    - operator: str, the inequality operator (">", "<", ">=", "<=")
    - right_value: float, the value on the right side of the inequality (e.g., 0)

    Returns:
    - str, the solution in interval notation
    """
    try:
        # Replace '^' with '**' for SymPy compatibility in left expression
        left_expr = add_multiplication_sign(left_expr.replace('^', '**').replace(' ', ''))

        # Construct the inequality string
        inequality_str = f"{left_expr} {operator} {right_value}"

        # Parse the inequality string into a SymPy relational object
        transformations = standard_transformations + (implicit_multiplication_application,)
        inequality = parse_expr(inequality_str, transformations=transformations)

        # Solve the inequality
        solution = solve_univariate_inequality(inequality, x, relational=False)

        # Format the solution
        if isinstance(solution, sympy.Union):
            # Multiple intervals
            formatted_intervals = [format_interval(interval) for interval in solution.args]
            solution_str = ' U '.join(formatted_intervals)
        elif isinstance(solution, sympy.Interval):
            # Single interval
            solution_str = format_interval(solution)
        elif isinstance(solution, sympy.FiniteSet):
            # Finite set of points
            points = sorted(solution)
            formatted_points = ', '.join([f"x = {pt}" for pt in points])
            solution_str = formatted_points
        else:
            # No solution or other cases
            solution_str = "No solution."

        return solution_str

    except Exception as e:
        raise ValueError(f"Error in solving inequality: {str(e)}")

def Midpoint(x1, y1, x2, y2):
    # Convert the input points to Fraction to ensure rational results
    x1, y1, x2, y2 = Fraction(x1), Fraction(y1), Fraction(x2), Fraction(y2)

    # Calculate midpoint using fractions
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Calculate distance using symbolic sqrt (no simplification)
    distance = sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return (mid_x, mid_y), distance

def show_midpoint_distance():
    try:
        # Fetch the user input and convert it to floats first
        x1 = float(midpoint_x1_entry.get())
        y1 = float(midpoint_y1_entry.get())
        x2 = float(midpoint_x2_entry.get())
        y2 = float(midpoint_y2_entry.get())

        # Call Midpoint function
        midpoint, distance = Midpoint(x1, y1, x2, y2)

        # Display the result in a message box
        messagebox.showinfo("Midpoint and Distance",
                            f"Midpoint: ({midpoint[0]}, {midpoint[1]})\n"
                            f"Distance: {distance}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for the points.")

# Function to compute the Circle Equation and Radius using points
def Circle_Equation_from_points(center_x, center_y, point_x, point_y):
    # Convert the input points to Fraction to ensure rational results
    center_x, center_y = Fraction(center_x), Fraction(center_y)
    point_x, point_y = Fraction(point_x), Fraction(point_y)

    # Calculate the radius (distance between center and point on circumference)
    radius = sp.sqrt((point_x - center_x) ** 2 + (point_y - center_y) ** 2)

    # The equation of the circle in standard form: (x - h)^2 + (y - k)^2 = r^2
    h, k = center_x, center_y
    r_squared = radius ** 2

    # Standard form
    standard_form = f"(x - {h})^2 + (y - {k})^2 = {r_squared}"

    # General form: x^2 + y^2 + Dx + Ey + F = 0
    D = -2 * h
    E = -2 * k
    F = h**2 + k**2 - r_squared
    general_form = f"x^2 + y^2 + ({D})x + ({E})y + ({F}) = 0"

    return radius, standard_form, general_form

# Function to compute the center and radius from the given equation
def Circle_Equation_from_equation(equation):
    # Parse the equation to extract center and radius
    match = re.match(r"\(x - ([\d.-]+)\)\^2 \+ \(y - ([\d.-]+)\)\^2 = ([\d.-]+)", equation)
    if match:
        h = float(match.group(1))
        k = float(match.group(2))
        r_squared = float(match.group(3))
        radius = sp.sqrt(r_squared)

        center = (h, k)
        return center, radius
    else:
        raise ValueError("Invalid circle equation format. Please use the standard form: (x - h)^2 + (y - k)^2 = r^2.")

# GUI Logic to Display Circle Equation and Radius when points are given
def show_circle_equation_from_points():
    try:
        # Fetch the user input and convert it to floats first
        center_x = float(circle_center_x_entry.get())
        center_y = float(circle_center_y_entry.get())
        point_x = float(circle_point_x_entry.get())
        point_y = float(circle_point_y_entry.get())

        # Call Circle_Equation function
        radius, standard_form, general_form = Circle_Equation_from_points(center_x, center_y, point_x, point_y)

        # Display the result in a message box
        messagebox.showinfo("Circle Equation",
                            f"Radius: {radius}\n"
                            f"Standard Form: {standard_form}\n"
                            f"General Form: {general_form}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

# GUI Logic to Display Center and Radius when equation is given
def show_circle_equation_from_equation():
    try:
        # Fetch the equation input
        equation = circle_equation_entry.get()

        # Call Circle_Equation_from_equation function
        center, radius = Circle_Equation_from_equation(equation)

        # Display the result in a message box
        messagebox.showinfo("Circle Information",
                            f"Center: {center}\n"
                            f"Radius: {radius}")
    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))

# ------------------- Function Operations ------------------- #
def apply_function_operations(f_expr, g_expr):
    """Apply operations between two functions and return simplified algebraic expressions."""
    try:
        # Fix multiplication signs and exponentiation if needed
        f_expr = add_multiplication_sign(f_expr)
        g_expr = add_multiplication_sign(g_expr)

        # Convert expressions into algebraic form using sympy
        f = simplify(f_expr)
        g = simplify(g_expr)

        # Perform the operations symbolically
        f_plus_g = simplify(f + g)
        f_minus_g = simplify(f - g)
        f_times_g = expand(f * g)  # Expand to get the correct polynomial form
        f_div_g = simplify(f / g)

        # Find the domain where g(x) != 0 for division
        domain_f_div_g = solve(Eq(g, 0), x)
        domain_str = f"x != {domain_f_div_g}" if domain_f_div_g else "All real numbers"

        return {
            "f+g": f_plus_g,
            "f-g": f_minus_g,
            "f*g": f_times_g,
            "f/g": f_div_g,
            "f/g domain": domain_str
        }
    except Exception as e:
        raise ValueError(f"Error in function operations: {str(e)}")

def apply_difference_quotient(f_expr):
    """Calculates the difference quotient of a function."""
    try:
        # Fix multiplication signs and exponentiation if needed
        f_expr = add_multiplication_sign(f_expr)

        # Convert expression into algebraic form using sympy
        f = simplify(f_expr)

        # Compute f(x + h)
        f_x_plus_h = f.subs(x, x + h)

        # Compute the difference quotient: (f(x+h) - f(x)) / h
        difference_quotient = simplify((f_x_plus_h - f) / h)

        return difference_quotient
    except Exception as e:
        raise ValueError(f"Error in difference quotient calculation: {str(e)}")

# ------------------- GUI Logic ------------------- #
def show_result(choice, **kwargs):
    """Displays the result in a message box based on the problem type."""
    try:
        if choice == "Equation":
            equation = kwargs.get('equation')
            slope, y_intercept = calculate_slope_and_intercept(equation)
            if y_intercept is not None:
                messagebox.showinfo("Result", f"Slope: {slope}\nY-Intercept: {y_intercept}")
            else:
                messagebox.showinfo("Result", f"Slope: {slope}")
        elif choice == "Points":
            x1, y1, x2, y2 = kwargs.get('points')
            slope, y_intercept = calculate_slope_from_points(x1, y1, x2, y2)
            if y_intercept is not None:
                messagebox.showinfo("Result", f"Slope: {slope}\nY-Intercept: {y_intercept}")
            else:
                messagebox.showinfo("Result", f"Slope: {slope}")
        elif choice == "Parallel/Perpendicular":
            equation = kwargs.get('equation')
            x = kwargs.get('x')
            y = kwargs.get('y')
            relationship = kwargs.get('relationship')
            slope, y_intercept = calculate_slope_for_parallel_perpendicular(equation, x, y, relationship)
            if y_intercept is not None:
                messagebox.showinfo("Result", f"Slope: {slope}\nY-Intercept: {y_intercept}")
            else:
                messagebox.showinfo("Result", f"Slope: {slope}")
        elif choice == "System of Equations":
            eq1 = kwargs.get('eq1')
            eq2 = kwargs.get('eq2')
            eq3 = kwargs.get('eq3')
            method = kwargs.get('method')
            system_size = kwargs.get('system_size')

            if system_size == "2x2":
                if method == "addition":
                    x, y = addition_method(eq1, eq2)
                elif method == "substitution":
                    x, y = substitution_method(eq1, eq2)
                messagebox.showinfo("System of Equations Result", f"The solution is (x, y) = ({x:.2f}, {y:.2f})")
            elif system_size == "3x3":
                x, y, z = solve_3x3(eq1, eq2, eq3)
                messagebox.showinfo("System of Equations Result", f"The solution is (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f})")
        elif choice == "How_Many":
            price_adult = kwargs.get('price_adult')
            price_senior = kwargs.get('price_senior')
            total_people = kwargs.get('total_people')
            total_receipts = kwargs.get('total_receipts')

            # Solve the 'How Many' problem
            adults, seniors = solve_how_many(price_adult, price_senior, total_people, total_receipts)
            messagebox.showinfo("Result", f"Adults: {adults}\nSeniors: {seniors}")

        elif choice == "Absolute Value Equations":
            lhs_abs = kwargs.get('lhs_abs')
            lhs_non_abs = kwargs.get('lhs_non_abs')
            operator = kwargs.get('operator')
            rhs = kwargs.get('rhs')

            if not lhs_abs or rhs is None:
                raise ValueError("Please enter valid input.")

            # Solve the absolute value equation or inequality
            solution = solve_absolute_value_equation(lhs_abs, lhs_non_abs, operator, rhs)
            messagebox.showinfo("Result", f"Solution: {solution}")
        elif choice == "Domain and Range":
            points_str = kwargs.get('points')
            if not points_str:
                raise ValueError("Please enter a valid list of points.")

            domain, range_, is_function = domain_and_range(points_str)
            function_result = "True" if is_function else "False"
            messagebox.showinfo("Result", f"Domain: {domain}\nRange: {range_}\nFunction: {function_result}")
        elif choice == "F and G":
            f_expr = kwargs.get('f_expr')
            g_expr = kwargs.get('g_expr')
            if not f_expr or not g_expr:
                messagebox.showerror("Error", "Please enter valid functions for f(x) and g(x).")
                return

            results = apply_function_operations(f_expr, g_expr)
            result_str = (
                f"(f+g)(x) = {results['f+g']}\n"
                f"(f-g)(x) = {results['f-g']}\n"
                f"(f*g)(x) = {results['f*g']}\n"
                f"(f/g)(x) = {results['f/g']}\n"
                f"Domain for f/g: {results['f/g domain']}\n"
            )
            messagebox.showinfo("F and G Result", result_str)
        elif choice == "Difference Quotient":
            f_expr = kwargs.get('f_expr')
            if not f_expr:
                messagebox.showerror("Error", "Please enter a valid function for f(x).")
                return

            result = apply_difference_quotient(f_expr)
            result_str = f"Difference Quotient: {result}"
            messagebox.showinfo("Difference Quotient Result", result_str)
        elif choice == "Trig Functions from One Function":
            known_func = kwargs.get('known_func')
            known_value = kwargs.get('known_value')
            quadrant = kwargs.get('quadrant')
            if not known_func or not known_value or not quadrant:
                raise ValueError("Please enter all required fields.")
            results = calculate_remaining_trig_functions(known_func, known_value, quadrant)
            result_str = (
                f"Given {known_func}(θ) = {known_value}, θ in Quadrant {quadrant}\n\n"
                f"sin(θ) = {results['sin']}\n"
                f"cos(θ) = {results['cos']}\n"
                f"tan(θ) = {results['tan']}\n"
                f"csc(θ) = {results['csc']}\n"
                f"sec(θ) = {results['sec']}\n"
                f"cot(θ) = {results['cot']}\n"
                "\n(Simplify your answer, including any radicals. Use integers or fractions for any numbers in the expression.)"
            )
            messagebox.showinfo("Trig Functions Result", result_str)
        elif choice == "Synthetic Division":
            polynomial = kwargs.get('polynomial')
            divisor = kwargs.get('divisor')
            if not polynomial or not divisor:
                raise ValueError("Please enter both the polynomial and the divisor.")

            # Perform synthetic division
            quotient, remainder = synthetic_division(polynomial, divisor)

            # Display the result
            messagebox.showinfo(
                "Synthetic Division Result",
                f"Quotient: {quotient}\nRemainder: {remainder}"
            )
        else:
            pass
    except Exception as e:
        messagebox.showerror("Error", str(e))

def switch_system_size(*args):
    """Switches between 2x2 and 3x3 system input fields."""
    # Clear previous entries and frames
    eq1_entry.delete(0, tk.END)
    eq2_entry.delete(0, tk.END)
    eq3_entry.delete(0, tk.END)

    # Show or hide the third equation entry based on the selection
    if system_size_var.get() == "2x2":
        eq3_frame.pack_forget()
        method_menu.pack(pady=5)  # Show method menu only for 2 equations
    else:
        eq3_frame.pack(pady=10)
        method_menu.pack_forget()  # Hide method menu for 3 equations

def on_submit():
    """Handles the submission of user inputs based on the selected problem type."""
    problem_type = problem_var.get()

    if problem_type == "System of Equations":
        try:
            eq1 = eq1_entry.get()
            eq2 = eq2_entry.get()
            system_size = system_size_var.get()

            if system_size == "2x2":
                method = system_method_var.get()
                if method == "addition":
                    x, y = addition_method(eq1, eq2)
                elif method == "substitution":
                    x, y = substitution_method(eq1, eq2)
                messagebox.showinfo(
                    "System of Equations Result",
                    f"The solution is (x, y) = ({x:.2f}, {y:.2f})"
                )
            elif system_size == "3x3":
                eq3 = eq3_entry.get()
                x, y, z = solve_3x3(eq1, eq2, eq3)
                messagebox.showinfo(
                    "System of Equations Result",
                    f"The solution is (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f})"
                )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Plot Equation":
        try:
            input_type = var.get()

            if input_type == "Equation":
                equation = equation_entry.get()
                slope, y_intercept = calculate_slope_and_intercept(equation)
                if y_intercept is not None:
                    messagebox.showinfo(
                        "Slope of the Line Result",
                        f"Slope: {slope}\nY-Intercept: {y_intercept}"
                    )
                else:
                    messagebox.showinfo(
                        "Slope of the Line Result",
                        f"Slope: {slope}"
                    )
            elif input_type == "Points":
                x1 = float(x1_entry.get())
                y1 = float(y1_entry.get())
                x2 = float(x2_entry.get())
                y2 = float(y2_entry.get())
                slope, y_intercept = calculate_slope_from_points(x1, y1, x2, y2)
                if y_intercept is not None:
                    messagebox.showinfo(
                        "Slope of the Line Result",
                        f"Slope: {slope}\nY-Intercept: {y_intercept}"
                    )
                else:
                    messagebox.showinfo(
                        "Slope of the Line Result",
                        f"Slope: {slope}"
                    )
            elif input_type == "Parallel/Perpendicular":
                equation = pp_equation_entry.get()
                x = float(pp_x_entry.get())
                y = float(pp_y_entry.get())
                relationship = relationship_var.get()
                slope, y_intercept = calculate_slope_for_parallel_perpendicular(
                    equation, x, y, relationship
                )
                if y_intercept is not None:
                    messagebox.showinfo(
                        "Slope of the Line Result",
                        f"Slope: {slope}\nY-Intercept: {y_intercept}"
                    )
                else:
                    messagebox.showinfo(
                        "Slope of the Line Result",
                        f"Slope: {slope}"
                    )
            elif input_type == "Polynomials":
                try:
                    polynomial_str = polynomial_entry.get()
                    if not polynomial_str:
                        raise ValueError("Please enter a polynomial function.")

                    # Solve the polynomial to get factored form and roots
                    factored_form, solutions = solve_polynomial(polynomial_str)
                    solutions_str = ', '.join(str(sol) for sol in solutions)

                    # Check if the polynomial is quadratic
                    poly = Poly(validate_expression(polynomial_str), x)
                    if poly.degree() == 2:
                        # Analyze the quadratic function
                        quadratic_analysis = solve_quadratics(polynomial_str)

                        # Extract the entire analysis for plotting
                        result_str = (
                            f"Factored Form: {factored_form}\n"
                            f"Solutions (roots): {solutions_str}\n\n"
                            f"Domain: {quadratic_analysis['domain']}\n"
                            f"Range: {quadratic_analysis['range']}\n\n"
                            f"Intervals of Increase: {quadratic_analysis['increasing_interval']}\n"
                            f"Intervals of Decrease: {quadratic_analysis['decreasing_interval']}\n\n"
                            f"Concavity: {quadratic_analysis['concavity']}\n"
                            f"Axis of Symmetry: {quadratic_analysis['axis_of_symmetry']}\n"
                            f"Vertex: ({quadratic_analysis['vertex'][0]}, {quadratic_analysis['vertex'][1]})\n"
                            f"Y-Intercept: {quadratic_analysis['y_intercept']}\n"
                            f"X-Intercept(s): {', '.join(map(str, quadratic_analysis['x_intercepts']))}\n"
                            "\nThe quadratic function has been graphed."
                        )
                    else:
                        # For non-quadratic polynomials, provide only factored form and roots
                        result_str = (
                            f"Factored Form: {factored_form}\n"
                            f"Solutions (roots): {solutions_str}\n"
                            "The polynomial has been graphed."
                        )

                    # Display the result in a message box
                    messagebox.showinfo(
                        "Polynomial Result",
                        result_str
                    )

                    # Plot the polynomial
                    if poly.degree() == 2:
                        plot_quadratic(validate_expression(polynomial_str), quadratic_analysis)
                    else:
                        plot_polynomial(validate_expression(polynomial_str), solutions)

                except Exception as e:
                    messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "How_Many":
        try:
            price_adult = float(adult_price_entry.get())
            price_senior = float(senior_price_entry.get())
            total_people = int(total_people_entry.get())
            total_receipts = float(total_receipts_entry.get())

            # Solve the 'How Many' problem
            adults, seniors = solve_how_many(
                price_adult, price_senior, total_people, total_receipts
            )
            messagebox.showinfo(
                "How Many Result",
                f"Adults: {adults}\nSeniors: {seniors}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Absolute Value Equations":
        try:
            lhs_abs = abs_lhs_entry.get()
            lhs_non_abs = non_abs_lhs_entry.get()
            operator = abs_operator_var.get()
            rhs = abs_rhs_entry.get()

            solution = solve_absolute_value_equation(
                lhs_abs, lhs_non_abs, operator, rhs
            )
            messagebox.showinfo(
                "Absolute Value Equation Result",
                f"Solution: {solution}"
            )
        except NotImplementedError as nie:
            messagebox.showerror("Not Implemented", str(nie))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Domain and Range":
        try:
            points_str = domain_range_points_entry.get()
            domain, range_, is_function = domain_and_range(points_str)
            function_result = "True" if is_function else "False"
            messagebox.showinfo(
                "Domain and Range Result",
                f"Domain: {domain}\nRange: {range_}\nFunction: {function_result}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "F and G":
        try:
            f_expr = f_expr_entry.get()
            g_expr = g_expr_entry.get()
            if not f_expr or not g_expr:
                messagebox.showerror(
                    "Error",
                    "Please enter valid functions for f(x) and g(x)."
                )
                return

            results = apply_function_operations(f_expr, g_expr)
            result_str = (
                f"(f+g)(x) = {results['f+g']}\n"
                f"(f-g)(x) = {results['f-g']}\n"
                f"(f*g)(x) = {results['f*g']}\n"
                f"(f/g)(x) = {results['f/g']}\n"
                f"Domain for f/g: {results['f/g domain']}\n"
            )
            messagebox.showinfo("F and G Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Difference Quotient":
        try:
            f_expr = dq_f_expr_entry.get()
            if not f_expr:
                messagebox.showerror(
                    "Error",
                    "Please enter a valid function for f(x)."
                )
                return

            result = apply_difference_quotient(f_expr)
            result_str = f"Difference Quotient: {result}"
            messagebox.showinfo("Difference Quotient Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Degrees and Radians":
        try:
            angle_value = degrees_radians_entry.get()
            input_unit = degrees_radians_unit_var.get()  # "Degrees" or "Radians"
            result = degrees_radians_conversion(angle_value, input_unit)
            if input_unit == "Degrees":
                messagebox.showinfo("Degrees to Radians Result", f"{angle_value} degrees equals {result} radians")
            else:
                messagebox.showinfo("Radians to Degrees Result", f"{angle_value} radians equals {result} degrees")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Radians to DMS":
        try:
            angle_value = radians_to_dms_entry.get()
            decimal_degrees, degrees_part, minutes, seconds = radians_to_dms(angle_value)
            result_str = (
                f"{angle_value} radians equals:\n"
                f"{decimal_degrees} degrees (decimal degrees rounded to 4 decimal places)\n"
                f"or\n"
                f"{degrees_part} degrees {minutes}' {seconds}'' (DMS format)"
            )
            messagebox.showinfo("Radians to DMS Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Arc Length":
        try:
            radius = arc_radius_entry.get()
            angle_degrees = arc_angle_entry.get()
            if not radius or not angle_degrees:
                raise ValueError("Please enter both radius and central angle.")
            exact_arc_length, approximate_arc_length = calculate_arc_length(radius, angle_degrees)
            result_str = (
                f"Exact arc length = {exact_arc_length} units\n"
                f"Approximate arc length = {approximate_arc_length} units"
            )
            messagebox.showinfo("Arc Length Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Find Velocity":
        try:
            radius = velocity_radius_entry.get()
            rev_per_min = velocity_rpm_entry.get()
            if not radius or not rev_per_min:
                raise ValueError("Please enter both the radius and the rotational speed (rev/min).")
            angular_velocity, linear_velocity = calculate_velocity(radius, rev_per_min)
            # Round angular velocity to nearest hundredth and linear velocity to nearest whole number
            angular_velocity_rounded = round(float(angular_velocity), 2)
            linear_velocity_rounded = int(round(float(linear_velocity)))
            result_str = (
                f"Angular velocity ω = {angular_velocity_rounded} rad/sec\n"
                f"Linear velocity v = {linear_velocity_rounded} units/sec"
            )
            messagebox.showinfo("Find Velocity Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Quadrants":
        try:
            # Import pi from sympy
            from sympy import pi, simplify

            quadrant = int(quadrant_entry.get())
            if quadrant not in [1, 2, 3, 4]:
                raise ValueError("Please enter a quadrant number between 1 and 4.")
            trig_values = calculate_trig_values(quadrant)
            result_str = "Exact Trig Values for Quadrant {}\n\n".format(quadrant)
            for item in trig_values:
                angle_deg = item['theta'] * 180 / pi
                angle_str = simplify(angle_deg)
                result_str += (
                    f"Theta: {item['theta']} radians ({angle_str} degrees)\n"
                    f"sin(theta) = {item['sin']}\n"
                    f"cos(theta) = {item['cos']}\n"
                    f"tan(theta) = {item['tan']}\n"
                    f"csc(theta) = {item['csc']}\n"
                    f"sec(theta) = {item['sec']}\n"
                    f"cot(theta) = {item['cot']}\n\n"
                )
            messagebox.showinfo("Quadrants Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Trig 6":
        try:
            # Import necessary functions
            from sympy import pi, simplify, S

            angle_str = trig6_angle_entry.get()
            unit = trig6_unit_var.get()  # "Degrees" or "Radians"

            if not angle_str:
                raise ValueError("Please enter an angle.")

            trig_values = calculate_trig_functions(angle_str, unit)

            angle_display = angle_str + (" degrees" if unit == "Degrees" else " radians")

            result_str = f"Trigonometric Functions for {angle_display}:\n\n"
            result_str += f"sin(theta) = {trig_values['sin']}\n"
            result_str += f"cos(theta) = {trig_values['cos']}\n"
            result_str += f"tan(theta) = {trig_values['tan']}\n"
            result_str += f"csc(theta) = {trig_values['csc']}\n"
            result_str += f"sec(theta) = {trig_values['sec']}\n"
            result_str += f"cot(theta) = {trig_values['cot']}\n"

            messagebox.showinfo("Trig 6 Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Trig Ratios (x,y)":
        try:
            x_value = trig_ratios_x_entry.get()
            y_value = trig_ratios_y_entry.get()
            if not x_value or not y_value:
                raise ValueError("Please enter both x and y coordinates.")
            trig_ratios = calculate_trig_ratios_xy(x_value, y_value)
            result_str = (
                f"Given P({x_value}, {y_value}), r = {trig_ratios['r']}\n\n"
                f"sin(θ) = {trig_ratios['sin']}\n"
                f"cos(θ) = {trig_ratios['cos']}\n"
                f"tan(θ) = {trig_ratios['tan']}\n"
                f"csc(θ) = {trig_ratios['csc']}\n"
                f"sec(θ) = {trig_ratios['sec']}\n"
                f"cot(θ) = {trig_ratios['cot']}\n"
                "\n(Simplify your answer. Type an exact answer, using radicals as needed. "
                "Use integers or fractions for any numbers in the expression.)"
            )
            messagebox.showinfo("Trig Ratios Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Trig Functions from One Function":
        try:
            known_func = trig_one_func_var.get()
            known_value = trig_one_value_entry.get()
            quadrant = trig_one_quadrant_entry.get()
            if not known_func or not known_value or not quadrant:
                raise ValueError("Please enter all required fields.")
            results = calculate_remaining_trig_functions(known_func, known_value, quadrant)
            result_str = (
                f"Given {known_func}(θ) = {known_value}, θ in Quadrant {quadrant}\n\n"
                f"sin(θ) = {results['sin']}\n"
                f"cos(θ) = {results['cos']}\n"
                f"tan(θ) = {results['tan']}\n"
                f"csc(θ) = {results['csc']}\n"
                f"sec(θ) = {results['sec']}\n"
                f"cot(θ) = {results['cot']}\n"
                "\n(Simplify your answer, including any radicals. Use integers or fractions for any numbers in the expression.)"
            )
            messagebox.showinfo("Trig Functions Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Amplitude and Period":
        try:
            equation_str = amplitude_period_entry.get()
            if not equation_str:
                raise ValueError("Please enter a trigonometric function.")
            # Capture all four returned values: amplitude, period, phase shift, and key points
            amplitude, period, phase_shift, key_points = calculate_amplitude_and_period(equation_str)
            result_str = (
                f"Amplitude: {amplitude}\n"
                f"Period: {period}\n"
                f"Phase Shift: {phase_shift} radians\n"
                f"Key Points: {key_points}\n"
                "The function has been graphed."
            )
            messagebox.showinfo("Amplitude and Period Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Piecewise Functions":
        try:
            is_defined = piecewise_defined_var.get()
            if is_defined == "Undefined":
                expr1 = piecewise_expr1_entry.get()
                expr2 = piecewise_expr2_entry.get()
                expr3 = piecewise_expr3_entry.get()
                x_value = piecewise_x_entry.get()

                if not (expr1 and expr2 and expr3 and x_value):
                    raise ValueError("Please enter all the required fields.")

                # Convert x_value to float
                x_val = float(x_value)

                # Evaluate the piecewise function
                result = evaluate_piecewise_function_undefined(expr1, expr2, expr3, x_val)

                messagebox.showinfo("Piecewise Function Result", f"f({x_val}) = {result}")
            else:
                expr1 = piecewise_cond_expr1_entry.get()
                op1 = piecewise_op1_var.get()
                val1 = piecewise_val1_entry.get()
                expr2 = piecewise_cond_expr2_entry.get()
                op2 = piecewise_op2_var.get()
                val2 = piecewise_val2_entry.get()

                if not (expr1 and op1 and val1 and expr2 and op2 and val2):
                    raise ValueError("Please enter all the required fields.")

                # Evaluate and plot the piecewise function
                result = evaluate_piecewise_function_defined(expr1, op1, val1, expr2, op2, val2)

                messagebox.showinfo("Piecewise Function Result", result)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Solution Set of Equations":
        try:
            lhs = equation_lhs_entry.get()
            rhs = equation_rhs_entry.get()
            solutions = solve_equation(lhs, rhs)
            messagebox.showinfo("Equation Solution", f"Solution(s): {solutions}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Solve Quadratics":
        try:
            equation_str = quadratic_equation_entry.get()
            if not equation_str:
                raise ValueError("Please enter a quadratic equation.")
            results = solve_quadratics(equation_str)
            vertex = results['vertex']
            axis_of_symmetry = results['axis_of_symmetry']
            concavity = results['concavity']
            x_intercepts = results['x_intercepts']
            y_intercept = results['y_intercept']

            # Format x-intercepts for display
            x_intercepts_str = ', '.join(str(xi) for xi in x_intercepts)

            result_str = (
                f"Vertex: ({vertex[0]}, {vertex[1]})\n"
                f"Axis of Symmetry: {axis_of_symmetry}\n"
                f"Concavity: {concavity}\n"
                f"X-Intercept(s): {x_intercepts_str}\n"
                f"Y-Intercept: {y_intercept}\n"
                "The quadratic function has been graphed."
            )
            messagebox.showinfo("Quadratic Function Result", result_str)

            # Plot the quadratic function
            plot_quadratic(validate_expression(equation_str), results)
        except Exception as e:
            # Print traceback to console for debugging
            traceback.print_exc()
            # Show a messagebox with the error
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    elif problem_type == "NegArc":
        try:
            function_name = neg_arc_func_var.get()
            mode = neg_arc_mode_var.get()
            value_str = neg_arc_value_entry.get()
            if not function_name or not mode or not value_str:
                raise ValueError("Please enter all required fields.")
            result = neg_arc_function(function_name, mode, value_str)
            if result == "No solution":
                messagebox.showinfo("NegArc Result", "No solution")
            else:
                angle_rad, angle_deg = result
                message = f"The angle in radians: {angle_rad}\nThe angle in degrees: {angle_deg}"
                messagebox.showinfo("NegArc Result", message)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Right Triangle":
        try:
            # Collect known values
            selected_vars = {var_name: var.get() for var_name, var in known_vars.items()}
            num_selected = sum(selected_vars.values())

            if num_selected != 2:
                raise ValueError("Please select exactly two known values.")

            given_values = {}
            for var_name, is_selected in selected_vars.items():
                if is_selected:
                    value_str = entries[var_name].get()
                    if not value_str:
                        raise ValueError(f"Please enter a value for {var_name}.")
                    given_values[var_name] = float(value_str)

            # Call the right_triangle function
            result = right_triangle(given_values)

            # Format the result for display
            result_str = (
                f"a = {result['a']}\n"
                f"b = {result['b']}\n"
                f"c = {result['c']}\n"
                f"A = {result['A']} degrees\n"
                f"B = {result['B']} degrees\n"
                f"C = {result['C']} degrees\n"
            )
            messagebox.showinfo("Right Triangle Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Solve Trig Equations in Intervals":
        try:
            equation_str = trig_intervals_equation_entry.get()
            interval_unit = trig_intervals_unit_var.get()

            if not equation_str or not interval_unit:
                raise ValueError("Please enter both the equation and select the interval unit.")

            # Solve the equation
            result = solve_trig_intervals(equation_str, interval_unit)

            # Use the correct keys from the result dictionary
            transformed_eq = result['transformed_eq']
            solutions = result['solutions']

            if not solutions:
                messagebox.showinfo(
                    "Trig Intervals Result",
                    f"Transformed Equation:\n{transformed_eq}\n\nNo solutions found in the specified interval."
                )
            else:
                # Join solutions into a single string with line breaks
                solutions_str = '\n'.join(solutions)
                messagebox.showinfo(
                    "Trig Intervals Result",
                    f"Transformed Equation:\n{transformed_eq}\n\nSolutions:\n{solutions_str}"
                )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Sinusoidal Function":
        try:
            # Get the input values from the GUI as fractions or floats
            y_intercept = safe_fraction_or_float(sinusoidal_y_intercept_entry.get())
            trough_y = safe_fraction_or_float(sinusoidal_trough_y_entry.get())
            trough_x = safe_fraction_or_float(sinusoidal_trough_x_entry.get())
            first_wave_x = safe_fraction_or_float(sinusoidal_first_wave_x_entry.get())

            # Call the function to calculate the sinusoidal equation
            equation = find_sinusoidal_equation(y_intercept, trough_y, trough_x, first_wave_x)

            # Display the result in a message box
            messagebox.showinfo("Sinusoidal Function Result", f"The sinusoidal equation is: y = {equation}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Synthetic Division":
        try:
            polynomial = synthetic_poly_entry.get()
            divisor = synthetic_divisor_entry.get()
            if not polynomial or not divisor:
                raise ValueError("Please enter both the polynomial and the divisor.")

            # Perform polynomial division
            quotient, remainder = divide_polynomials(polynomial, divisor)

            # Display the quotient and remainder
            messagebox.showinfo(
                "Polynomial Division Result",
                f"Quotient: {quotient}\nRemainder: {remainder}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Rational Zeros":
        try:
            polynomial = rational_zeros_entry.get()
            if not polynomial:
                raise ValueError("Please enter a polynomial function.")

            # Get the potential zeros, actual zeros, remaining zeros, and factored form
            results = rational_zeros(polynomial)
            potential_zeros = results['potential_zeros']
            actual_rational_zeros = results['actual_rational_zeros']
            remaining_zeros = results['remaining_zeros']
            factored_form = results['factored_form']

            # Format the potential zeros
            if potential_zeros:
                potential_zeros_str = ', '.join([str(zero) for zero in potential_zeros])
            else:
                potential_zeros_str = "No potential rational zeros found."

            # Format the actual rational zeros
            if actual_rational_zeros:
                actual_zeros_str = ', '.join([str(zero) for zero in actual_rational_zeros])
            else:
                actual_zeros_str = "No actual rational zeros found."

            # Format the remaining zeros (irrational or complex)
            if remaining_zeros:
                # Simplify the remaining zeros for display
                remaining_zeros_str = ', '.join([str(zero) for zero in remaining_zeros])
            else:
                remaining_zeros_str = "No remaining zeros found."

            # Format the factored form
            if actual_rational_zeros or remaining_zeros:
                # Construct the factored form string
                factors = []
                for zero in actual_rational_zeros:
                    if zero < 0:
                        factors.append(f"(x + {abs(zero)})")
                    else:
                        factors.append(f"(x - {zero})")
                for rem_zero in remaining_zeros:
                    if rem_zero.is_real:
                        if rem_zero < 0:
                            factors.append(f"(x + {abs(rem_zero)})")
                        else:
                            factors.append(f"(x - {rem_zero})")
                    else:
                        # For complex zeros, include them as conjugate pairs
                        conj_zero = rem_zero.conjugate()
                        factors.append(f"(x - {rem_zero})")
                        factors.append(f"(x - {conj_zero})")
                factored_form_str = "P(x) = " + "*".join(factors)
            else:
                factored_form_str = "Cannot factor the polynomial with the given zeros."

            # Prepare the result message
            result_message = (
                f"Potential Rational Zeros:\n{potential_zeros_str}\n\n"
                f"Actual Rational Zeros Found:\n{actual_zeros_str}\n\n"
                f"Remaining Zeros (Irrational or Complex):\n{remaining_zeros_str}\n\n"
                f"Factored Form of the Polynomial:\n{factored_form_str}"
            )

            # Display the result in a message box
            messagebox.showinfo("Rational Zeros Result", result_message)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Solve Inequalities":
        try:
            # Retrieve inputs from the GUI
            left_expr = solve_inequalities_left_entry.get()
            operator = inequality_operator_var.get()
            right_value = solve_inequalities_right_entry.get()

            # Input validation
            if not left_expr:
                raise ValueError("Please enter the left expression.")
            if operator not in [">", "<", ">=", "<="]:
                raise ValueError("Please select a valid inequality operator.")
            if not right_value:
                raise ValueError("Please enter the right value.")

            # Attempt to convert right_value to a number (int or float)
            try:
                right_value_numeric = float(right_value)
            except ValueError:
                raise ValueError("Right value must be a number.")

            # Solve the inequality
            solution = solve_inequalities(left_expr, operator, right_value_numeric)

            # Prepare the result message
            result_message = (
                f"The solution is:\n{solution}\n\n"
                f"(Type your answer in interval notation. Simplify your answer. "
                f"Use integers or fractions for any numbers in the expression.)"
            )

            # Display the solution
            messagebox.showinfo("Solve Inequalities Result", result_message)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "Rational Functions":
        try:
            rational_str = rational_entry.get()
            if not rational_str:
                raise ValueError("Please enter a rational function.")
            analysis = analyze_rational_function(rational_str)

            # Prepare the result string
            result_str = ""

            # Asymptotes
            if analysis["vertical_asymptotes"]:
                vas = ', '.join([f"x = {va}" for va in analysis["vertical_asymptotes"]])
                result_str += f"Vertical Asymptote(s): {vas}\n"
            else:
                result_str += "No Vertical Asymptotes.\n"

            if analysis["horizontal_asymptote"] is not None:
                result_str += f"Horizontal Asymptote: y = {analysis['horizontal_asymptote']}\n"
            elif analysis["oblique_asymptote"] is not None:
                result_str += f"Oblique Asymptote: y = {analysis['oblique_asymptote']}\n"
            else:
                result_str += "No Horizontal or Oblique Asymptotes.\n"

            # Intercepts
            if analysis["x_intercepts"]:
                xis = ', '.join([f"({xi}, 0)" for xi in analysis["x_intercepts"]])
                result_str += f"X-Intercept(s): {xis}\n"
            else:
                result_str += "No X-Intercepts.\n"

            if analysis["y_intercept"] is not None:
                # Convert y_intercept to a fraction string using SymPy's pretty printing
                y_intercept_frac = sympy.nsimplify(analysis["y_intercept"], rational=True)
                result_str += f"Y-Intercept: (0, {y_intercept_frac})\n"
            else:
                result_str += "Y-Intercept: Undefined (Denominator is zero at x = 0).\n"

            # Holes
            if analysis["holes"]:
                holes_str = ""
                for idx, hole in enumerate(analysis["holes"], start=1):
                    x_hole, y_hole = hole
                    if y_hole is not None:
                        holes_str += f"Hole {idx}: ({x_hole}, {y_hole})\n"
                    else:
                        holes_str += f"Hole {idx}: (Undefined y-value)\n"
                result_str += f"Hole(s):\n{holes_str}"
            else:
                result_str += "No Holes in the Graph.\n"

            result_str += "The rational function has been graphed."

            # Display the result
            messagebox.showinfo(
                "Rational Function Result",
                result_str
            )

            # Plot the rational function
            plot_rational_function(rational_str, analysis)
        except Exception as e:
            # Print traceback to console for debugging
            traceback.print_exc()
            # Show a messagebox with the error
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    elif problem_type == "Midpoint and Distance":
        show_midpoint_distance()  # Call midpoint and distance function
    #elif problem_type == "Circle Equation":
        #input_type = circle_input_type_var.get()

        #if input_type == "Points":
            #show_circle_equation_from_points()  # Call function for points input
        #elif input_type == "Equation":
            #show_circle_equation_from_equation()  # Call function for equation input
        #else:
            #messagebox.showerror("Error", "Invalid input type selected.")
    else:
        messagebox.showerror(
            "Error",
            "Please select a math problem type."
        )

def switch_input_type(*args):
    """Switches between different input frames based on the selected input type."""
    # Hide all frames initially
    equation_frame.pack_forget()
    points_frame.pack_forget()
    parallel_perpendicular_frame.pack_forget()
    polynomial_frame.pack_forget()  # Hide the polynomial frame initially
    #circle_frame.pack_forget()
    #circle_points_frame.pack_forget()
    #circle_equation_frame.pack_forget()

    # Show the relevant frame based on the selected input type
    choice = var.get()
    if choice == "Equation":
        equation_frame.pack(pady=10)
    elif choice == "Points":
        points_frame.pack(pady=10)
    elif choice == "Parallel/Perpendicular":
        parallel_perpendicular_frame.pack(pady=10)
    elif choice == "Polynomials":
        polynomial_frame.pack(pady=10)
    #elif choice == "Circle":
        #choice = circle_input_type_var.get()
        #if choice == "Points":
            #circle_points_frame.pack(pady=10)
        #elif choice == "Equation":
            #circle_equation_frame.pack(pady=10)
    else:
        pass

def switch_problem_type(*args):
    """Switches the visible input frames based on the selected problem type."""
    # Hide all frames initially
    input_type_menu.pack_forget()
    equation_frame.pack_forget()
    points_frame.pack_forget()
    parallel_perpendicular_frame.pack_forget()
    system_frame.pack_forget()
    how_many_frame.pack_forget()
    absolute_value_frame.pack_forget()
    domain_range_frame.pack_forget()
    fg_frame.pack_forget()
    dq_frame.pack_forget()
    degrees_radians_frame.pack_forget()
    radians_to_dms_frame.pack_forget()
    arc_length_frame.pack_forget()
    find_velocity_frame.pack_forget()
    quadrants_frame.pack_forget()
    trig6_frame.pack_forget()
    trig_ratios_frame.pack_forget()
    trig_one_func_frame.pack_forget()
    amplitude_period_frame.pack_forget()
    piecewise_frame.pack_forget()
    solution_equation_frame.pack_forget()
    quadratic_frame.pack_forget()
    neg_arc_frame.pack_forget()
    right_triangle_frame.pack_forget()
    trig_intervals_frame.pack_forget()
    sinusoidal_frame.pack_forget()
    synthetic_division_frame.pack_forget()
    rational_zeros_frame.pack_forget()
    solve_inequalities_frame.pack_forget()
    rational_frame.pack_forget()
    midpoint_frame.pack_forget()
    #circle_frame.pack_forget()

    # Show the relevant frame based on the selected problem type
    problem_type = problem_var.get()
    if problem_type == "Plot Equation":
        var.set("Equation")  # Default to "Equation" input type
        input_type_menu.pack(pady=10)
        switch_input_type()  # Show input type frame based on the dropdown
    elif problem_type == "System of Equations":
        system_frame.pack(pady=10)  # Directly show the system frame
    elif problem_type == "How_Many":
        how_many_frame.pack(pady=10)  # Show the How_Many input frame
    elif problem_type == "Absolute Value Equations":
        switch_absolute_value_input()  # Show the absolute value equation frame
    elif problem_type == "Domain and Range":
        domain_range_frame.pack(pady=10)
    elif problem_type == "F and G":
        fg_frame.pack(pady=10)  # Show the F and G input frame
    elif problem_type == "Difference Quotient":
        dq_frame.pack(pady=10)  # Show the Difference Quotient input frame
    elif problem_type == "Degrees and Radians":
        degrees_radians_frame.pack(pady=10)
    elif problem_type == "Radians to DMS":
        radians_to_dms_frame.pack(pady=10)
    elif problem_type == "Arc Length":
        arc_length_frame.pack(pady=10)
    elif problem_type == "Find Velocity":
        find_velocity_frame.pack(pady=10)
    elif problem_type == "Quadrants":
        quadrants_frame.pack(pady=10)
    elif problem_type == "Trig 6":
        trig6_frame.pack(pady=10)
    elif problem_type == "Trig Ratios (x,y)":
        trig_ratios_frame.pack(pady=10)
    elif problem_type == "Trig Functions from One Function":
        trig_one_func_frame.pack(pady=10)
    elif problem_type == "Amplitude and Period":
        amplitude_period_frame.pack(pady=10)
    elif problem_type == "Piecewise Functions":
        piecewise_frame.pack(pady=10)
        switch_piecewise_mode()  # Ensure correct fields are shown
    elif problem_type == "Solution Set of Equations":
        solution_equation_frame.pack(pady=10)
    elif problem_type == "Solve Quadratics":
        quadratic_frame.pack(pady=10)
    elif problem_type == "NegArc":
        neg_arc_frame.pack(pady=10)
    elif problem_type == "Right Triangle":
        right_triangle_frame.pack(pady=10)
    elif problem_type == "Solve Trig Equations in Intervals":
        trig_intervals_frame.pack(pady=10)
    elif problem_type == "Sinusoidal Function":
        sinusoidal_frame.pack(pady=10)
    elif problem_type == "Synthetic Division":
        synthetic_division_frame.pack(pady=10)
    elif problem_type == "Rational Zeros":
        rational_zeros_frame.pack(pady=10)
    elif problem_type == "Solve Inequalities":
        solve_inequalities_frame.pack(pady=10)
    elif problem_type == "Rational Functions":
        rational_frame.pack(pady=10)
    elif problem_type == "Midpoint and Distance":
        midpoint_frame.pack(pady=10)
    #elif problem_type == "Circle Equation":
        #circle_frame.pack(pady=10)
        #pass
    else:
        messagebox.showerror("Error", "Invalid problem type selected.")

def switch_absolute_value_input(*args):
    """Switches to the Absolute Value Equation input frame."""
    # Hide other frames and show absolute value frame
    equation_frame.pack_forget()
    points_frame.pack_forget()
    parallel_perpendicular_frame.pack_forget()
    system_frame.pack_forget()
    how_many_frame.pack_forget()
    absolute_value_frame.pack(pady=10)
    domain_range_frame.pack_forget()
    fg_frame.pack_forget()
    dq_frame.pack_forget()

def switch_piecewise_mode(*args):
    """Switches the input fields based on whether the function is defined or undefined outside intervals."""
    if piecewise_defined_var.get() == "Undefined":
        # Show the entries for three expressions and x value
        expr1_label.pack()
        piecewise_expr1_entry.pack()
        expr2_label.pack()
        piecewise_expr2_entry.pack()
        expr3_label.pack()
        piecewise_expr3_entry.pack()
        x_value_label.pack()
        piecewise_x_entry.pack()

        # Hide the conditional expressions and operators
        cond_expr1_label.pack_forget()
        piecewise_cond_expr1_entry.pack_forget()
        op1_label.pack_forget()
        piecewise_op1_menu.pack_forget()
        val1_label.pack_forget()
        piecewise_val1_entry.pack_forget()

        cond_expr2_label.pack_forget()
        piecewise_cond_expr2_entry.pack_forget()
        op2_label.pack_forget()
        piecewise_op2_menu.pack_forget()
        val2_label.pack_forget()
        piecewise_val2_entry.pack_forget()
    else:
        # Show the entries for conditional expressions
        cond_expr1_label.pack()
        piecewise_cond_expr1_entry.pack()
        op1_label.pack()
        piecewise_op1_menu.pack()
        val1_label.pack()
        piecewise_val1_entry.pack()

        cond_expr2_label.pack()
        piecewise_cond_expr2_entry.pack()
        op2_label.pack()
        piecewise_op2_menu.pack()
        val2_label.pack()
        piecewise_val2_entry.pack()

        # Hide the entries for three expressions and x value
        expr1_label.pack_forget()
        piecewise_expr1_entry.pack_forget()
        expr2_label.pack_forget()
        piecewise_expr2_entry.pack_forget()
        expr3_label.pack_forget()
        piecewise_expr3_entry.pack_forget()
        x_value_label.pack_forget()
        piecewise_x_entry.pack_forget()

# ------------------- GUI Setup ------------------- #
# Creating the main window
window = tk.Tk()
window.title("Math Function Solver")
window.geometry("500x700")

# Dropdown for choosing the math problem type
problem_var = tk.StringVar(window)
problem_var.set("Select a Math Problem")

problem_type_menu = tk.OptionMenu(window, problem_var,
                                  "Plot Equation",
                                  "System of Equations",
                                  "How_Many",
                                  "Absolute Value Equations",
                                  "Domain and Range",
                                  "F and G",
                                  "Difference Quotient",
                                  "Degrees and Radians",
                                  "Radians to DMS",
                                  "Arc Length",
                                  "Find Velocity",
                                  "Quadrants",
                                  "Trig 6",
                                  "Trig Ratios (x,y)",
                                  "Trig Functions from One Function",
                                  "Amplitude and Period",
                                  "Piecewise Functions",
                                  "Solution Set of Equations",
                                  "Solve Quadratics",
                                  "NegArc",
                                  "Right Triangle",
                                  "Solve Trig Equations in Intervals",
                                  "Sinusoidal Function",
                                  "Synthetic Division",
                                  "Rational Zeros",
                                  "Solve Inequalities",
                                  "Rational Functions",
                                  "Midpoint and Distance",
                                  #"Circle Equation",
                                  command=switch_problem_type)
problem_type_menu.pack(pady=10)

# Define the variable for the input type of the circle (Points or Equation)
#circle_input_type_var = tk.StringVar(window)
#circle_input_type_var.set("Points")  # Default to Points input type

# Set the dropdown to trigger this function
#circle_input_type_menu = tk.OptionMenu(window, circle_input_type_var, "Points", "Equation", command=switch_input_type)
#circle_input_type_menu.pack(pady=10)

# Frame for Points input (initially hidden)
#circle_points_frame = tk.Frame(window)
#tk.Label(circle_points_frame, text="Enter the center of the circle (center_x, center_y):").pack()
#circle_center_x_entry = tk.Entry(circle_points_frame, width=10)
#circle_center_y_entry = tk.Entry(circle_points_frame, width=10)
#circle_center_x_entry.pack(side="left", padx=5)
#circle_center_y_entry.pack(side="left", padx=5)

#tk.Label(circle_points_frame, text="Enter a point on the circumference (point_x, point_y):").pack()
#circle_point_x_entry = tk.Entry(circle_points_frame, width=10)
#circle_point_y_entry = tk.Entry(circle_points_frame, width=10)
#circle_point_x_entry.pack(side="left", padx=5)
#circle_point_y_entry.pack(side="left", padx=5)

# Frame for Equation input (initially hidden)
#circle_equation_frame = tk.Frame(window)
#tk.Label(circle_equation_frame, text="Enter the circle equation in standard form (e.g., (x - 3)^2 + (y + 2)^2 = 25):").pack()
#circle_equation_entry = tk.Entry(circle_equation_frame, width=50)
#circle_equation_entry.pack(pady=10)

# Initially hide all circle-related frames
#circle_points_frame.pack_forget()
#circle_equation_frame.pack_forget()

# Frame for Midpoint and Distance input
midpoint_frame = tk.Frame(window)
tk.Label(midpoint_frame, text="Enter the first point (x1, y1):").pack()
midpoint_x1_entry = tk.Entry(midpoint_frame, width=10)
midpoint_y1_entry = tk.Entry(midpoint_frame, width=10)
midpoint_x1_entry.pack(side="left", padx=5)
midpoint_y1_entry.pack(side="left", padx=5)

tk.Label(midpoint_frame, text="Enter the second point (x2, y2):").pack()
midpoint_x2_entry = tk.Entry(midpoint_frame, width=10)
midpoint_y2_entry = tk.Entry(midpoint_frame, width=10)
midpoint_x2_entry.pack(side="left", padx=5)
midpoint_y2_entry.pack(side="left", padx=5)

# Frame for Rational Function Analysis
rational_frame = tk.Frame(window)
tk.Label(rational_frame, text="Enter a rational function (e.g., 1/(x + 3)):", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
rational_entry = tk.Entry(rational_frame, width=50)
rational_entry.grid(row=1, column=0, padx=5, pady=5)


# 2. GUI Frame for Solve Inequalities
solve_inequalities_frame = tk.Frame(window)

# Label for the frame
tk.Label(solve_inequalities_frame, text="Solve Inequalities", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)

# Entry for Left Expression
tk.Label(solve_inequalities_frame, text="Left Expression:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
solve_inequalities_left_entry = tk.Entry(solve_inequalities_frame, width=30)
solve_inequalities_left_entry.grid(row=1, column=1, padx=5, pady=5)
solve_inequalities_left_entry.insert(0, "(x-3)*(x-4)*(x-5)")

# Dropdown for Operator
tk.Label(solve_inequalities_frame, text="Operator:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
inequality_operator_var = tk.StringVar(solve_inequalities_frame)
inequality_operator_var.set(">=")  # Default operator
inequality_operator_menu = tk.OptionMenu(solve_inequalities_frame, inequality_operator_var, ">", "<", ">=", "<=")
inequality_operator_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')

# Entry for Right Value
tk.Label(solve_inequalities_frame, text="Right Value:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
solve_inequalities_right_entry = tk.Entry(solve_inequalities_frame, width=30)
solve_inequalities_right_entry.grid(row=3, column=1, padx=5, pady=5)
solve_inequalities_right_entry.insert(0, "0")

# Frame for Rational Zeros input
rational_zeros_frame = tk.Frame(window)

tk.Label(rational_zeros_frame, text="Enter the polynomial (e.g., x^5 -16x^2 +10x -29):").pack(pady=5)
rational_zeros_entry = tk.Entry(rational_zeros_frame, width=50)
rational_zeros_entry.pack(pady=5)



# Synthetic Division Frame
synthetic_division_frame = tk.Frame(window)

tk.Label(synthetic_division_frame, text="Enter the polynomial (e.g., 2x^3 - 6x^2 + 2x -1):").pack(pady=5)
synthetic_poly_entry = tk.Entry(synthetic_division_frame, width=50)
synthetic_poly_entry.pack(pady=5)

tk.Label(synthetic_division_frame, text="Enter the divisor (e.g., x - 3):").pack(pady=5)
synthetic_divisor_entry = tk.Entry(synthetic_division_frame, width=30)
synthetic_divisor_entry.pack(pady=5)

# Frame for Sinusoidal Function input
sinusoidal_frame = tk.Frame(window)

tk.Label(sinusoidal_frame, text="Enter the y-intercept at x = 0:").pack()
sinusoidal_y_intercept_entry = tk.Entry(sinusoidal_frame, width=30)
sinusoidal_y_intercept_entry.pack()

tk.Label(sinusoidal_frame, text="Enter the y-value at the trough:").pack()
sinusoidal_trough_y_entry = tk.Entry(sinusoidal_frame, width=30)
sinusoidal_trough_y_entry.pack()

tk.Label(sinusoidal_frame, text="Enter the x-value at the trough:").pack()
sinusoidal_trough_x_entry = tk.Entry(sinusoidal_frame, width=30)
sinusoidal_trough_x_entry.pack()

tk.Label(sinusoidal_frame, text="Enter the x-value at the end of the first wave:").pack()
sinusoidal_first_wave_x_entry = tk.Entry(sinusoidal_frame, width=30)
sinusoidal_first_wave_x_entry.pack()


# Frame for Trig Intervals input
trig_intervals_frame = tk.Frame(window)

tk.Label(trig_intervals_frame, text="Enter the trigonometric equation (e.g., 2*sin(x) - sqrt(2) = 0):").pack()
trig_intervals_equation_entry = tk.Entry(trig_intervals_frame, width=50)
trig_intervals_equation_entry.pack()

tk.Label(trig_intervals_frame, text="Select the interval unit:").pack()
# Updated code
trig_intervals_unit_var = tk.StringVar(window)
trig_intervals_unit_var.set("Radians")  # Default to Radians
trig_intervals_unit_menu = tk.OptionMenu(
    trig_intervals_frame, trig_intervals_unit_var, "Radians", "Degrees", "Degrees (0 to 180)"
)
trig_intervals_unit_menu.pack()



# Frame for Right Triangle input
right_triangle_frame = tk.Frame(window)

# Instructions
tk.Label(right_triangle_frame, text="Select any two known values and enter their values.").pack()

# Variables for checkboxes
known_vars = {'a': tk.IntVar(), 'b': tk.IntVar(), 'c': tk.IntVar(),
              'A': tk.IntVar(), 'B': tk.IntVar()}

# Create checkboxes and entries
entries = {}

for var_name in ['a', 'b', 'c', 'A', 'B']:
    frame = tk.Frame(right_triangle_frame)
    var_check = tk.Checkbutton(frame, text=f"{var_name}", variable=known_vars[var_name])
    var_check.pack(side="left")
    entry = tk.Entry(frame, width=10)
    entries[var_name] = entry
    entry.pack(side="left")
    frame.pack(anchor="w")

tk.Label(right_triangle_frame, text="(Note: Angle C is always 90 degrees)").pack()


# Frame for NegArc input
neg_arc_frame = tk.Frame(window)

tk.Label(neg_arc_frame, text="Select the trigonometric function:").pack()
neg_arc_func_var = tk.StringVar(window)
neg_arc_func_var.set("sin")  # Default to 'sin'
neg_arc_func_menu = tk.OptionMenu(neg_arc_frame, neg_arc_func_var, "sin", "cos", "tan", "csc", "sec", "cot")
neg_arc_func_menu.pack()

tk.Label(neg_arc_frame, text="Select the mode:").pack()
neg_arc_mode_var = tk.StringVar(window)
neg_arc_mode_var.set("-1")  # Default to '-1'
neg_arc_mode_menu = tk.OptionMenu(neg_arc_frame, neg_arc_mode_var, "-1", "arc")
neg_arc_mode_menu.pack()

tk.Label(neg_arc_frame, text="Enter the value (e.g., 1, sqrt(3)/2):").pack()
neg_arc_value_entry = tk.Entry(neg_arc_frame, width=30)
neg_arc_value_entry.pack()


# Frame for Quadratic Equation input
quadratic_frame = tk.Frame(window)
tk.Label(quadratic_frame, text="Enter the quadratic expression (e.g., -(x-2)^2+9):").pack()
quadratic_equation_entry = tk.Entry(quadratic_frame, width=40)
quadratic_equation_entry.pack()

# Frame for Solution Set of Equations input
solution_equation_frame = tk.Frame(window)
tk.Label(solution_equation_frame, text="Enter the left-hand side of the equation (e.g., x^2 + 2x - 35):").pack()
equation_lhs_entry = tk.Entry(solution_equation_frame, width=40)
equation_lhs_entry.pack()
tk.Label(solution_equation_frame, text="Enter the right-hand side of the equation (e.g., 0):").pack()
equation_rhs_entry = tk.Entry(solution_equation_frame, width=40)
equation_rhs_entry.pack()

# Frame for Piecewise Functions input
piecewise_frame = tk.Frame(window)

# Ask if the function is defined or undefined outside the given intervals
tk.Label(piecewise_frame, text="Is the function defined outside the given intervals?").pack()
piecewise_defined_var = tk.StringVar(value="Undefined")
tk.Radiobutton(piecewise_frame, text="Defined", variable=piecewise_defined_var, value="Defined", command=switch_piecewise_mode).pack()
tk.Radiobutton(piecewise_frame, text="Undefined", variable=piecewise_defined_var, value="Undefined", command=switch_piecewise_mode).pack()

# For Undefined case (three expressions and x value)
expr1_label = tk.Label(piecewise_frame, text="Enter the expression for f(x) when x < 0:")
piecewise_expr1_entry = tk.Entry(piecewise_frame, width=30)

expr2_label = tk.Label(piecewise_frame, text="Enter the expression for f(x) when x = 0:")
piecewise_expr2_entry = tk.Entry(piecewise_frame, width=30)

expr3_label = tk.Label(piecewise_frame, text="Enter the expression for f(x) when x > 0:")
piecewise_expr3_entry = tk.Entry(piecewise_frame, width=30)

x_value_label = tk.Label(piecewise_frame, text="Enter the x value to evaluate (e.g., -1):")
piecewise_x_entry = tk.Entry(piecewise_frame, width=30)

# For Defined case (conditional expressions)
cond_expr1_label = tk.Label(piecewise_frame, text="Enter the expression for the first piece:")
piecewise_cond_expr1_entry = tk.Entry(piecewise_frame, width=30)

op1_label = tk.Label(piecewise_frame, text="Select the operator for the first piece:")
piecewise_op1_var = tk.StringVar(value="<=")
piecewise_op1_menu = tk.OptionMenu(piecewise_frame, piecewise_op1_var, "<", "<=", ">", ">=", "==")

val1_label = tk.Label(piecewise_frame, text="Enter the value for the first piece condition (e.g., -1):")
piecewise_val1_entry = tk.Entry(piecewise_frame, width=10)

cond_expr2_label = tk.Label(piecewise_frame, text="Enter the expression for the second piece:")
piecewise_cond_expr2_entry = tk.Entry(piecewise_frame, width=30)

op2_label = tk.Label(piecewise_frame, text="Select the operator for the second piece:")
piecewise_op2_var = tk.StringVar(value=">")
piecewise_op2_menu = tk.OptionMenu(piecewise_frame, piecewise_op2_var, "<", "<=", ">", ">=", "==")

val2_label = tk.Label(piecewise_frame, text="Enter the value for the second piece condition (e.g., -1):")
piecewise_val2_entry = tk.Entry(piecewise_frame, width=10)

# Initially set the correct fields based on default value
switch_piecewise_mode()

# Dropdown for choosing input type (only for "Slope of a Line")
var = tk.StringVar(window)
var.set("Equation")

input_type_menu = tk.OptionMenu(window, var, "Equation", "Points", "Parallel/Perpendicular", "Polynomials", command=switch_input_type)

# Create the polynomial input frame
polynomial_frame = tk.Frame(window)
tk.Label(polynomial_frame, text="Enter a polynomial function (e.g., x^3 - 4x^2 - 7x + 10):").pack()
polynomial_entry = tk.Entry(polynomial_frame, width=40)
polynomial_entry.pack()

# Frame for Quadrants input
quadrants_frame = tk.Frame(window)
tk.Label(quadrants_frame, text="Enter the quadrant number (1-4):").pack()
quadrant_entry = tk.Entry(quadrants_frame, width=10)
quadrant_entry.pack()

# Frame for Trig Functions from One Function input
trig_one_func_frame = tk.Frame(window)
tk.Label(trig_one_func_frame, text="Select the known trigonometric function:").pack()
trig_one_func_var = tk.StringVar(window)
trig_one_func_var.set("cos")  # Default to 'cos'
trig_one_func_menu = tk.OptionMenu(trig_one_func_frame, trig_one_func_var, "sin", "cos", "tan", "csc", "sec", "cot")
trig_one_func_menu.pack()

tk.Label(trig_one_func_frame, text="Enter the value of the known function (e.g., -4/5):").pack()
trig_one_value_entry = tk.Entry(trig_one_func_frame, width=30)
trig_one_value_entry.pack()

tk.Label(trig_one_func_frame, text="Enter the quadrant number (1-4):").pack()
trig_one_quadrant_entry = tk.Entry(trig_one_func_frame, width=10)
trig_one_quadrant_entry.pack()

# Create a new frame for the Trig Functions from One Function input
trig_one_func_frame = tk.Frame(window)
tk.Label(trig_one_func_frame, text="Select the known trigonometric function:").pack()
trig_one_func_var = tk.StringVar(window)
trig_one_func_var.set("cos")  # Default to 'cos'
trig_one_func_menu = tk.OptionMenu(trig_one_func_frame, trig_one_func_var, "sin", "cos", "tan", "csc", "sec", "cot")
trig_one_func_menu.pack()

tk.Label(trig_one_func_frame, text="Enter the value of the known function (e.g., -4/5):").pack()
trig_one_value_entry = tk.Entry(trig_one_func_frame, width=30)
trig_one_value_entry.pack()

tk.Label(trig_one_func_frame, text="Enter the quadrant number (1-4):").pack()
trig_one_quadrant_entry = tk.Entry(trig_one_func_frame, width=10)
trig_one_quadrant_entry.pack()

# Frame for Amplitude and Period input
amplitude_period_frame = tk.Frame(window)
tk.Label(amplitude_period_frame, text="Enter the trigonometric function (e.g., y = 5 cos(1/7x)) :").pack()
amplitude_period_entry = tk.Entry(amplitude_period_frame, width=40)
amplitude_period_entry.pack()

# Frame for Trig Ratios (x,y) input
trig_ratios_frame = tk.Frame(window)
tk.Label(trig_ratios_frame, text="Enter the x-coordinate:").pack()
trig_ratios_x_entry = tk.Entry(trig_ratios_frame, width=30)
trig_ratios_x_entry.pack()

tk.Label(trig_ratios_frame, text="Enter the y-coordinate:").pack()
trig_ratios_y_entry = tk.Entry(trig_ratios_frame, width=30)
trig_ratios_y_entry.pack()

# Frame for Trig 6 input
trig6_frame = tk.Frame(window)
tk.Label(trig6_frame, text="Enter the angle (e.g., pi/3, 45):").pack()
trig6_angle_entry = tk.Entry(trig6_frame, width=30)
trig6_angle_entry.pack()

tk.Label(trig6_frame, text="Select the unit:").pack()
trig6_unit_var = tk.StringVar(window)
trig6_unit_var.set("Radians")  # Default to Radians
trig6_unit_menu = tk.OptionMenu(trig6_frame, trig6_unit_var, "Degrees", "Radians")
trig6_unit_menu.pack()


# Frame for Find Velocity input
find_velocity_frame = tk.Frame(window)
tk.Label(find_velocity_frame, text="Enter the radius (units):").pack()
velocity_radius_entry = tk.Entry(find_velocity_frame, width=30)
velocity_radius_entry.pack()

tk.Label(find_velocity_frame, text="Enter the rotational speed (rev/min):").pack()
velocity_rpm_entry = tk.Entry(find_velocity_frame, width=30)
velocity_rpm_entry.pack()


# Frame for Arc Length input
arc_length_frame = tk.Frame(window)
tk.Label(arc_length_frame, text="Enter the radius of the circle:").pack()
arc_radius_entry = tk.Entry(arc_length_frame, width=30)
arc_radius_entry.pack()

tk.Label(arc_length_frame, text="Enter the central angle in degrees:").pack()
arc_angle_entry = tk.Entry(arc_length_frame, width=30)
arc_angle_entry.pack()


# Frame for domain and range input
domain_range_frame = tk.Frame(window)
tk.Label(domain_range_frame, text="Enter the set of points (e.g., {(9,5)(21,-10)(31,5)}):").pack()
domain_range_points_entry = tk.Entry(domain_range_frame, width=40)
domain_range_points_entry.pack()

# Frame for Difference Quotient input
dq_frame = tk.Frame(window)
tk.Label(dq_frame, text="Enter f(x):").pack()
dq_f_expr_entry = tk.Entry(dq_frame, width=30)
dq_f_expr_entry.pack()

# Frame for F and G input
fg_frame = tk.Frame(window)
tk.Label(fg_frame, text="Enter f(x):").pack()
f_expr_entry = tk.Entry(fg_frame, width=30)
f_expr_entry.pack()

tk.Label(fg_frame, text="Enter g(x):").pack()
g_expr_entry = tk.Entry(fg_frame, width=30)
g_expr_entry.pack()

# Frame for Degrees and Radians input
degrees_radians_frame = tk.Frame(window)
tk.Label(degrees_radians_frame, text="Enter the angle value:").pack()
degrees_radians_entry = tk.Entry(degrees_radians_frame, width=30)
degrees_radians_entry.pack()

tk.Label(degrees_radians_frame, text="Select the input unit:").pack()
degrees_radians_unit_var = tk.StringVar(window)
degrees_radians_unit_var.set("Degrees")  # Default to Degrees
degrees_radians_unit_menu = tk.OptionMenu(degrees_radians_frame, degrees_radians_unit_var, "Degrees", "Radians")
degrees_radians_unit_menu.pack()


# Frame for absolute value equations input
absolute_value_frame = tk.Frame(window)
tk.Label(absolute_value_frame, text="Enter the absolute value portion (e.g., 2x+8):").pack()
abs_lhs_entry = tk.Entry(absolute_value_frame, width=30)
abs_lhs_entry.pack()

tk.Label(absolute_value_frame, text="Enter the non-absolute value portion (optional, e.g., -7):").pack()
non_abs_lhs_entry = tk.Entry(absolute_value_frame, width=30)
non_abs_lhs_entry.pack()

tk.Label(absolute_value_frame, text="Select the operator:").pack()
abs_operator_var = tk.StringVar(window)
abs_operator_var.set("=")
abs_operator_menu = tk.OptionMenu(absolute_value_frame, abs_operator_var, "=", ">", "<", ">=", "<=")
abs_operator_menu.pack()

tk.Label(absolute_value_frame, text="Enter the right-hand side value (e.g., 14):").pack()
abs_rhs_entry = tk.Entry(absolute_value_frame, width=30)
abs_rhs_entry.pack()



# Frame for Radians to DMS input
radians_to_dms_frame = tk.Frame(window)
tk.Label(radians_to_dms_frame, text="Enter the angle in radians:").pack()
radians_to_dms_entry = tk.Entry(radians_to_dms_frame, width=30)
radians_to_dms_entry.pack()


# Frame for equation input
equation_frame = tk.Frame(window)
tk.Label(equation_frame, text="Enter a linear equation (e.g., 5x - 4y = 20):").pack()
equation_entry = tk.Entry(equation_frame, width=30)
equation_entry.pack()

# Frame for points input
points_frame = tk.Frame(window)
tk.Label(points_frame, text="Enter the first point (x1, y1):").pack()
x1_entry = tk.Entry(points_frame, width=10)
y1_entry = tk.Entry(points_frame, width=10)
x1_entry.pack(side="left", padx=5)
y1_entry.pack(side="left", padx=5)

tk.Label(points_frame, text="Enter the second point (x2, y2):").pack()
x2_entry = tk.Entry(points_frame, width=10)
y2_entry = tk.Entry(points_frame, width=10)
x2_entry.pack(side="left", padx=5)
y2_entry.pack(side="left", padx=5)

# Frame for parallel/perpendicular input
parallel_perpendicular_frame = tk.Frame(window)
tk.Label(parallel_perpendicular_frame, text="Enter a reference equation (e.g., 5x - 4y = 20):").pack()
pp_equation_entry = tk.Entry(parallel_perpendicular_frame, width=30)
pp_equation_entry.pack()

tk.Label(parallel_perpendicular_frame, text="Enter a point (x, y):").pack()
pp_x_entry = tk.Entry(parallel_perpendicular_frame, width=10)
pp_y_entry = tk.Entry(parallel_perpendicular_frame, width=10)
pp_x_entry.pack(side="left", padx=5)
pp_y_entry.pack(side="left", padx=5)

tk.Label(parallel_perpendicular_frame, text="Select relationship:").pack()
relationship_var = tk.StringVar(window)
relationship_var.set("parallel")
relationship_menu = tk.OptionMenu(parallel_perpendicular_frame, relationship_var, "parallel", "perpendicular")
relationship_menu.pack()

# Frame for system of equations input
system_frame = tk.Frame(window)

# Dropdown to select between 2x2 and 3x3 systems
tk.Label(system_frame, text="Select the system type:").pack()
system_size_var = tk.StringVar(window)
system_size_var.set("2x2")  # Default to 2x2 system
system_size_menu = tk.OptionMenu(system_frame, system_size_var, "2x2", "3x3", command=switch_system_size)
system_size_menu.pack(pady=5)

# Input fields for the first two equations (shown for both 2x2 and 3x3 systems)
tk.Label(system_frame, text="Enter the first equation (e.g., y = 2x + 1 or 6y - 6x = 13):").pack()
eq1_entry = tk.Entry(system_frame, width=30)
eq1_entry.pack()

tk.Label(system_frame, text="Enter the second equation (e.g., y = -3x + 5 or 4y + 2x = 8):").pack()
eq2_entry = tk.Entry(system_frame, width=30)
eq2_entry.pack()

# Input field for the third equation (only shown for 3x3 systems)
eq3_frame = tk.Frame(system_frame)
tk.Label(eq3_frame, text="Enter the third equation (for 3x3 system):").pack()
eq3_entry = tk.Entry(eq3_frame, width=30)
eq3_entry.pack()

# Dropdown for choosing the method (only for 2x2 systems)
tk.Label(system_frame, text="Choose the method:").pack()
system_method_var = tk.StringVar(window)
system_method_var.set("addition")
method_menu = tk.OptionMenu(system_frame, system_method_var, "addition", "substitution")

# Add the input fields for 'How_Many'
how_many_frame = tk.Frame(window)

tk.Label(how_many_frame, text="Price per adult:").pack()
adult_price_entry = tk.Entry(how_many_frame, width=30)
adult_price_entry.pack()

tk.Label(how_many_frame, text="Price per senior:").pack()
senior_price_entry = tk.Entry(how_many_frame, width=30)
senior_price_entry.pack()

tk.Label(how_many_frame, text="Total number of people:").pack()
total_people_entry = tk.Entry(how_many_frame, width=30)
total_people_entry.pack()

tk.Label(how_many_frame, text="Total receipts:").pack()
total_receipts_entry = tk.Entry(how_many_frame, width=30)
total_receipts_entry.pack()

# Submit button
submit_button = tk.Button(window, text="Submit", command=on_submit)
submit_button.pack(pady=20)

# Run the Tkinter main loop

window.mainloop()
