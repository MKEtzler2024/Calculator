import tkinter as tk
from tkinter import messagebox
import re
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
from sympy import symbols, simplify, solve, Eq, expand

# Define the symbols 'x' and 'h' for algebraic operations
x, h = symbols('x h')

# Function to extract coefficients from standard form and slope-intercept form equations
def extract_coefficients(equation):
    equation = equation.replace(" ", "")  # Remove spaces

    # Match patterns for standard form and slope-intercept form
    match_standard = re.match(r"([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)y=([-+]?\d+\.?\d*)", equation)
    match_reverse_standard = re.match(r"([-+]?\d*\.?\d*)y([-+]?\d*\.?\d*)x=([-+]?\d+\.?\d*)", equation)  # Handles equations like 3y - 7x = 9
    match_slope_intercept = re.match(r"y=([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)?", equation)

    if match_standard:
        A = match_standard.group(1)
        B = match_standard.group(2)
        C = match_standard.group(3)

        # Handle missing or empty coefficients properly
        A = float(A) if A not in ["", "+", "-"] else float(f"{A}1")
        B = float(B) if B not in ["", "+", "-"] else float(f"{B}1")
        C = float(C)

        return A, B, C

    elif match_reverse_standard:
        B = match_reverse_standard.group(1)
        A = match_reverse_standard.group(2)
        C = match_reverse_standard.group(3)

        A = float(A) if A not in ["", "+", "-"] else float(f"{A}1")
        B = float(B) if B not in ["", "+", "-"] else float(f"{B}1")
        C = float(C)

        return -A, B, C

    elif match_slope_intercept:
        m = match_slope_intercept.group(1)
        b = match_slope_intercept.group(2)

        m = float(m) if m not in ["", "+", "-"] else float(f"{m}1")
        b = float(b) if b else 0.0

        # Convert to standard form: y - mx = b => -mx + y = b
        A = -m
        B = 1
        C = b

        return A, B, C

    else:
        raise ValueError("Invalid equation format. Please use 'Ax + By = C' or 'y = mx + b'.")

def extract_coefficients_3x3(equation):
    equation = equation.replace(" ", "")  # Remove spaces

    # Match pattern for 3-variable standard form equations (Ax + By + Cz = D)
    match = re.match(r"([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)y([-+]?\d*\.?\d*)z=([-+]?\d+\.?\d*)", equation)

    if match:
        A = match.group(1)
        B = match.group(2)
        C = match.group(3)
        D = match.group(4)

        # Handle missing or empty coefficients properly
        A = float(A) if A not in ["", "+", "-"] else float(f"{A}1")
        B = float(B) if B not in ["", "+", "-"] else float(f"{B}1")
        C = float(C) if C not in ["", "+", "-"] else float(f"{C}1")
        D = float(D)

        return A, B, C, D
    else:
        raise ValueError("Invalid 3-variable equation format. Please use 'Ax + By + Cz = D'.")


# Function to calculate slope and y-intercept from an equation
def calculate_slope_and_intercept(equation):
    try:
        A, B, C = extract_coefficients(equation)
        if B == 0:
            slope = "Undefined (Vertical line)"
            x_intercept = C / A
            y_intercept = None
        else:
            slope = -A / B
            y_intercept = C / B
            x_intercept = C / A if A != 0 else None

        # Plot the line
        plot_line(slope, y_intercept, x_intercept)

        return slope, y_intercept
    except Exception as e:
        raise ValueError(str(e))

# Function to calculate slope and y-intercept from two points
def calculate_slope_from_points(x1, y1, x2, y2):
    if x1 == x2:
        slope = "Undefined (Vertical line)"
        x_intercept = x1
        y_intercept = None
    else:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
        x_intercept = -y_intercept / slope if slope != 0 else None

    # Plot the line
    plot_line(slope, y_intercept, x_intercept)

    return slope, y_intercept

# Function to calculate slope for parallel or perpendicular lines
def calculate_slope_for_parallel_perpendicular(equation, x, y, relationship):
    try:
        A, B, C = extract_coefficients(equation)
        if B == 0:
            original_slope = None  # Vertical line
        else:
            original_slope = -A / B

        if relationship == "parallel":
            slope = original_slope
        elif relationship == "perpendicular":
            if original_slope == 0:
                slope = "Undefined (Vertical line)"
            elif original_slope is None:
                slope = 0
            else:
                slope = -1 / original_slope
        else:
            raise ValueError("Relationship must be either 'parallel' or 'perpendicular'.")

        if slope == "Undefined (Vertical line)":
            y_intercept = None
            x_intercept = x
        else:
            y_intercept = y - slope * x
            x_intercept = -y_intercept / slope if slope != 0 else None

        # Plot the line
        plot_line(slope, y_intercept, x_intercept)

        return slope, y_intercept
    except Exception as e:
        raise ValueError(str(e))

# Function to plot the line
def plot_line(slope, y_intercept, x_intercept):
    plt.figure(figsize=(6, 6))
    x_vals = np.linspace(-10, 10, 400)

    if slope == "Undefined (Vertical line)":
        x = np.full_like(x_vals, x_intercept)
        y = x_vals
        plt.plot(x, y, '-r', label=f'x = {x_intercept:.2f}')
    else:
        y = slope * x_vals + y_intercept
        plt.plot(x_vals, y, '-r', label=f'y = {slope:.2f}x + {y_intercept:.2f}')

    plt.title('Graph of the Linear Equation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# Function to solve the system of equations using substitution
def substitution_method(eq1_str, eq2_str):
    # Manually extract coefficients for eq1 and eq2
    a1, b1, c1 = extract_coefficients(eq1_str)  # For eq1 (e.g., y = 3x + 1 or Ax + By = C)
    a2, b2, c2 = extract_coefficients(eq2_str)  # For eq2 (e.g., 3y - 5x = 11 or Ax + By = C)

    # Ensure the first equation is solved for y, otherwise rearrange it
    if b1 == 1:  # This means eq1 is already in the form y = mx + b or can be rearranged
        # Substitute the value of y from eq1 into eq2
        # eq2: a2 * x + b2 * y = c2 => a2 * x + b2 * (a1 * x + c1) = c2
        # Solve for x
        x = (c2 - b2 * c1) / (a2 + b2 * a1)
        # Now substitute x into eq1 to solve for y
        y = a1 * x + c1
    elif a1 != 0 and b1 != 0:  # If it's in standard form, rearrange to solve for y
        # From a1 * x + b1 * y = c1, solve for y: y = (c1 - a1 * x) / b1
        # Substitute this into eq2: a2 * x + b2 * ((c1 - a1 * x) / b1) = c2
        x = (c2 * b1 - b2 * c1) / (a2 * b1 - b2 * a1)
        y = (c1 - a1 * x) / b1
    else:
        raise ValueError("The first equation must be solvable for y or rearrangeable to y = mx + b form.")

    return round(-x, 4), round(y, 4)

# Addition (Elimination) method for any two linear equations
def addition_method(eq1_str, eq2_str):
    # Extract coefficients
    a1, b1, c1 = extract_coefficients(eq1_str)  # For eq1 (e.g., Ax + By = C)
    a2, b2, c2 = extract_coefficients(eq2_str)  # For eq2 (e.g., Ax + By = C)

    # We need to equalize either the coefficient of x or y by scaling the equations
    # We'll equalize x coefficients here
    if a1 != a2:
        factor = a2 / a1
        # Scale the first equation
        a1_scaled = a1 * factor
        b1_scaled = b1 * factor
        c1_scaled = c1 * factor
    else:
        # If coefficients are already equal, just use the original values
        a1_scaled, b1_scaled, c1_scaled = a1, b1, c1

    # Subtract the equations to eliminate x
    delta_b = b1_scaled - b2
    delta_c = c1_scaled - c2

    if delta_b != 0:
        # Solve for y
        y = delta_c / delta_b

        # Substitute y back into one of the original equations to solve for x
        if a1 != 0:
            x = (c1 - b1 * y) / a1
        elif a2 != 0:
            x = (c2 - b2 * y) / a2
    else:
        raise ValueError("No unique solution exists. The system may be inconsistent or dependent.")

    return x, y

def solve_3x3(eq1, eq2, eq3):
    # Extract coefficients for all three equations
    a1, b1, c1, d1 = extract_coefficients_3x3(eq1)
    a2, b2, c2, d2 = extract_coefficients_3x3(eq2)
    a3, b3, c3, d3 = extract_coefficients_3x3(eq3)

    # Set up the system as matrices
    A = np.array([[a1, b1, c1],
                  [a2, b2, c2],
                  [a3, b3, c3]])
    B = np.array([d1, d2, d3])

    # Solve using numpy
    try:
        solution = np.linalg.solve(A, B)
        return solution[0], solution[1], solution[2]  # Return x, y, z
    except np.linalg.LinAlgError:
        raise ValueError("The system has no unique solution (singular matrix).")

# Add the solve_how_many function to handle the 'How_Many' problem
def solve_how_many(price_adult, price_senior, total_people, total_receipts):
    try:
        # Set up the system of equations
        A = np.array([[price_adult, price_senior],
                      [1, 1]])
        B = np.array([total_receipts, total_people])

        # Solve the system using numpy
        solution = np.linalg.solve(A, B)
        return round(solution[0]), round(solution[1])  # Return the number of adults and seniors
    except np.linalg.LinAlgError:
        raise ValueError("The system has no unique solution.")

#solves absolute value equations
def solve_absolute_value_equation(lhs_abs, lhs_non_abs, operator, rhs):
    try:
        # Convert the RHS to a float
        rhs = float(rhs)

        # Convert the non-absolute value part (lhs_non_abs) to a float, default to 0 if left blank
        if lhs_non_abs == "" or lhs_non_abs is None:
            lhs_non_abs = 0.0
        else:
            lhs_non_abs = float(lhs_non_abs)

        # Separate the lhs_abs expression into parts (lhs_abs should be something like "2x+8")
        lhs_abs = lhs_abs.replace(" ", "")  # Remove spaces from the lhs
        pattern = re.compile(r'([-+]?\d*\.?\d*)x([-+]?\d+)?')
        match = pattern.match(lhs_abs)

        if match:
            A = match.group(1)
            B = match.group(2)

            A = float(A) if A not in ["", "+", "-"] else float(f"{A}1")
            B = float(B) if B else 0.0
        else:
            raise ValueError("Invalid format for the left-hand side inside the absolute value.")

        # Adjust the right-hand side based on the non-absolute part
        rhs -= lhs_non_abs

        def format_solution(value):
            """Helper function to return a fraction if not a whole number"""
            frac = Fraction(value).limit_denominator()
            return frac if frac.denominator != 1 else int(frac)

        # Solve based on the comparison type
        if operator == "=":
            # Solve two cases for |Ax + B| = rhs
            x1 = (rhs - B) / A
            x2 = (-rhs - B) / A
            return f"x = {format_solution(x1)} or x = {format_solution(x2)}"

        elif operator == ">":
            # Solve for |Ax + B| > rhs
            x1 = (rhs - B) / A
            x2 = (-rhs - B) / A
            return f"x > {format_solution(x1)} or x < {format_solution(x2)}"

        elif operator == "<":
            # Solve for |Ax + B| < rhs
            x1 = (rhs - B) / A
            x2 = (-rhs - B) / A
            return f"{format_solution(x2)} < x < {format_solution(x1)}"

        elif operator == ">=":
            # Solve for |Ax + B| >= rhs
            x1 = (rhs - B) / A
            x2 = (-rhs - B) / A
            return f"x >= {format_solution(x1)} or x <= {format_solution(x2)}"

        elif operator == "<=":
            # Solve for |Ax + B| <= rhs
            x1 = (rhs - B) / A
            x2 = (-rhs - B) / A
            return f"{format_solution(x2)} <= x <= {format_solution(x1)}"

    except Exception as e:
        raise ValueError(f"Error solving the absolute value equation or inequality: {str(e)}")

#solves domain and range of points
def domain_and_range(points_str):
    try:
        # Remove spaces and extract the list of points from the input string
        points_str = points_str.replace(" ", "").strip("{}")

        # Extract each point
        points = re.findall(r"\((-?\d+),(-?\d+)\)", points_str)
        points = [(int(x), int(y)) for x, y in points]

        # Calculate domain (set of all unique x-values) and sort them
        domain = sorted({x for x, y in points})

        # Calculate range (set of all unique y-values) and sort them
        range_ = sorted({y for x, y in points})

        # Check if it's a function (no x-value should repeat with different y-values)
        x_values = [x for x, y in points]
        function = len(x_values) == len(set(x_values))  # True if no repeated x-values

        return domain, range_, function
    except Exception as e:
        raise ValueError(f"Error processing the list of points: {str(e)}")

def add_multiplication_sign(expression):
    """Insert explicit multiplication signs between numbers and variables (like '3x' becomes '3*x') and replace ^ with ** for exponentiation."""
    # Replace 'x' with explicit multiplication and replace '^' with '**'
    expression = re.sub(r"(\d)(x)", r"\1*\2", expression)  # Add multiplication sign before variables
    expression = expression.replace("^", "**")  # Replace ^ with ** for exponentiation
    return expression

def is_whole_number(value, tol=1e-5):
    """Check if the value is a whole number within a small tolerance."""
    return np.isclose(value, round(value), atol=tol)

def apply_difference_quotient(f_expr):
    """Compute the difference quotient (f(x+h) - f(x)) / h."""
    try:
        # Fix multiplication signs and exponentiation if needed
        f_expr = add_multiplication_sign(f_expr)

        # Convert the expression into algebraic form using sympy
        f = simplify(f_expr)

        # Compute f(x+h)
        f_x_plus_h = f.subs(x, x + h)

        # Compute the difference quotient: (f(x+h) - f(x)) / h
        difference_quotient = simplify((f_x_plus_h - f) / h)

        return difference_quotient
    except Exception as e:
        raise ValueError(f"Error in difference quotient calculation: {str(e)}")


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


# Function to display results
def show_result(choice, **kwargs):
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
            method = kwargs.get('method')

            # Choose the method: either addition or substitution
            if method == "addition":
                x, y = addition_method(eq1, eq2)
            elif method == "substitution":
                x, y = substitution_method(eq1, eq2)
            else:
                raise ValueError("Invalid method selected.")

            messagebox.showinfo("Result", f"The solution is (x, y) = ({x:.2f}, {y:.2f})")
        elif choice == "How_Many":
            price_adult = kwargs.get('price_adult')
            price_senior = kwargs.get('price_senior')
            total_people = kwargs.get('total_people')
            total_receipts = kwargs.get('total_receipts')

            # Solve the 'How_Many' problem
            adults, seniors = solve_how_many(price_adult, price_senior, total_people, total_receipts)
            messagebox.showinfo("Result", f"Adults: {adults}\nSeniors: {seniors}")

        elif choice == "Absolute Value Equations":
            lhs_abs = kwargs.get('lhs_abs')
            lhs_non_abs = kwargs.get('lhs_non_abs')
            operator = kwargs.get('operator')
            rhs = kwargs.get('rhs')

            if not lhs_abs or not rhs:
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

        else:
            messagebox.showerror("Error", "Invalid choice.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Function to switch between 2 and 3 equation systems
def switch_system_size(*args):
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

# Function to handle user submission
def on_submit():
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
                messagebox.showinfo("System of Equations Result", f"The solution is (x, y) = ({x:.2f}, {y:.2f})")
            elif system_size == "3x3":
                eq3 = eq3_entry.get()
                x, y, z = solve_3x3(eq1, eq2, eq3)
                messagebox.showinfo("System of Equations Result", f"The solution is (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Absolute Value Equations":
        try:
            lhs_abs = abs_lhs_entry.get()
            lhs_non_abs = non_abs_lhs_entry.get()
            operator = abs_operator_var.get()
            rhs = abs_rhs_entry.get()

            solution = solve_absolute_value_equation(lhs_abs, lhs_non_abs, operator, rhs)
            messagebox.showinfo("Absolute Value Equation Result", f"Solution: {solution}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Domain and Range":
        try:
            points_str = domain_range_points_entry.get()
            domain, range_, is_function = domain_and_range(points_str)
            function_result = "True" if is_function else "False"
            messagebox.showinfo("Domain and Range Result", f"Domain: {domain}\nRange: {range_}\nFunction: {function_result}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "F and G":
        try:
            f_expr = f_expr_entry.get()
            g_expr = g_expr_entry.get()
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
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Difference Quotient":
        try:
            f_expr = dq_f_expr_entry.get()
            if not f_expr:
                messagebox.showerror("Error", "Please enter a valid function for f(x).")
                return

            result = apply_difference_quotient(f_expr)
            result_str = f"Difference Quotient: {result}"
            messagebox.showinfo("Difference Quotient Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    elif problem_type == "Slope of a Line":
        try:
            input_type = var.get()

            if input_type == "Equation":
                equation = equation_entry.get()
                slope, y_intercept = calculate_slope_and_intercept(equation)
                if y_intercept is not None:
                    messagebox.showinfo("Slope of the Line Result", f"Slope: {slope}\nY-Intercept: {y_intercept}")
                else:
                    messagebox.showinfo("Slope of the Line Result", f"Slope: {slope}")
            elif input_type == "Points":
                x1 = float(x1_entry.get())
                y1 = float(y1_entry.get())
                x2 = float(x2_entry.get())
                y2 = float(y2_entry.get())
                slope, y_intercept = calculate_slope_from_points(x1, y1, x2, y2)
                if y_intercept is not None:
                    messagebox.showinfo("Slope of the Line Result", f"Slope: {slope}\nY-Intercept: {y_intercept}")
                else:
                    messagebox.showinfo("Slope of the Line Result", f"Slope: {slope}")
            elif input_type == "Parallel/Perpendicular":
                equation = pp_equation_entry.get()
                x = float(pp_x_entry.get())
                y = float(pp_y_entry.get())
                relationship = relationship_var.get()
                slope, y_intercept = calculate_slope_for_parallel_perpendicular(equation, x, y, relationship)
                if y_intercept is not None:
                    messagebox.showinfo("Slope of the Line Result", f"Slope: {slope}\nY-Intercept: {y_intercept}")
                else:
                    messagebox.showinfo("Slope of the Line Result", f"Slope: {slope}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    elif problem_type == "How_Many":
        try:
            price_adult = float(adult_price_entry.get())
            price_senior = float(senior_price_entry.get())
            total_people = int(total_people_entry.get())
            total_receipts = float(total_receipts_entry.get())

            # Solve the 'How Many' problem
            adults, seniors = solve_how_many(price_adult, price_senior, total_people, total_receipts)
            messagebox.showinfo("How Many Result", f"Adults: {adults}\nSeniors: {seniors}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Function to switch input frames based on selection
def switch_input_type(*args):
    # Hide all frames initially
    equation_frame.pack_forget()
    points_frame.pack_forget()
    parallel_perpendicular_frame.pack_forget()
    system_frame.pack_forget()
    how_many_frame.pack_forget()

    # Show the relevant frame based on the selected input type
    choice = var.get()
    if choice == "Equation":
        equation_frame.pack(pady=10)
    elif choice == "Points":
        points_frame.pack(pady=10)
    elif choice == "Parallel/Perpendicular":
        parallel_perpendicular_frame.pack(pady=10)

# Function to switch between problem types
def switch_problem_type(*args):
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

    # Show the relevant frame based on the selected problem type
    problem_type = problem_var.get()
    if problem_type == "Slope of a Line":
        var.set("Equation")  # Default to "Equation" input type
        input_type_menu.pack(pady=10)
        switch_input_type()  # Show input type frame based on the dropdown
    elif problem_type == "System of Equations":
        system_frame.pack(pady=10)  # Directly show the system frame
    elif problem_type == "How_Many":
        how_many_frame.pack(pady=10)  # Show the How_Many input frame
    elif problem_type == "Absolute Value Equations":
        switch_absolute_value_input()  # Call the function to show the absolute value equation frame
    elif problem_type == "Domain and Range":
        domain_range_frame.pack(pady=10)
    elif problem_type == "F and G":
        fg_frame.pack(pady=10)  # Show the F and G input frame
    elif problem_type == "Difference Quotient":
        dq_frame.pack(pady=10)  # Show the Difference Quotient input frame

# Function to switch to Absolute Value Equation input frame
def switch_absolute_value_input(*args):
    # Hide other frames and show absolute value frame
    equation_frame.pack_forget()
    points_frame.pack_forget()
    parallel_perpendicular_frame.pack_forget()
    system_frame.pack_forget()
    how_many_frame.pack_forget()
    absolute_value_frame.pack(pady=10)

# Creating the main window
window = tk.Tk()
window.title("Math Function Solver")
window.geometry("400x500")

# Dropdown for choosing the math problem type
problem_var = tk.StringVar(window)
problem_var.set("Select a Math Problem")

problem_type_menu = tk.OptionMenu(window, problem_var, "Slope of a Line", "System of Equations", "How_Many", "Absolute Value Equations", "Domain and Range", "F and G", "Difference Quotient", command=switch_problem_type)
problem_type_menu.pack(pady=10)

# Dropdown for choosing input type
var = tk.StringVar(window)
var.set("Equation")

input_type_menu = tk.OptionMenu(window, var, "Equation", "Points", "Parallel/Perpendicular", command=switch_input_type)


# Frame for domain and range input
domain_range_frame = tk.Frame(window)
tk.Label(domain_range_frame, text="Enter the set of points (e.g., {(9,5)(21,-10)(31,5)}):").pack()
domain_range_points_entry = tk.Entry(domain_range_frame, width=30)
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

# Add the new input fields for 'How_Many'
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
