import os
import numpy as np
import pdb; 
import matplotlib.pyplot as plt
import csv
from sympy import symbols, solve, lambdify
from math import copysign

# Generate dataset for polynomials
np.random.seed(0)
num_samples = 1000  # Set the number of samples to 10 for each degree

# Define the base directory for the dataset
base_dir = "./Polynomial_Dataset"

# Check if the dataset directory exists; if not, create it
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

csv_file_path = os.path.join(base_dir, 'polynomial_labels.csv')

# Check if the CSV file already exists
csv_exists = os.path.isfile(csv_file_path)

# Updated coefficient range for better learning
coeff_range = 10

for degree in range(1, 5):
    

    for i in range(num_samples):
        if degree == 4 and i == 168:
            pdb.set_trace()
        # Pause the program on the target iteration
        coeffs = np.random.uniform(-2, 2, size=degree)
        # while all(-0.1 <= coeff <= 0.1 for coeff in coeffs) or not any(-2 <= coeff <= -0.1 or 0.1 <= coeff <= 2 for coeff in coeffs):
        #     coeffs = np.random.uniform(-2, 2, size=degree)
            
        coeffs_rounded = np.concatenate((np.zeros(4 - degree), np.round(coeffs, 2)))  # Pad with zeros and round coefficients

        # Define the symbolic variable and the polynomial expression
        x_sym = symbols('x')
        polynomial_expr = sum(coeff * x_sym ** (degree - j) for j, coeff in enumerate(coeffs))

        # Create a lambda function from the polynomial expression
        poly_func = lambdify(x_sym, polynomial_expr, 'numpy')

        # Solve for the roots
        roots = solve(polynomial_expr, x_sym)

        # Filter out complex roots and keep the real ones
        x_intercepts = [float(root.as_real_imag()[0]) for root in roots if root.is_real]

        # Identify the real critical points
        critical_real_points = [float(point.as_real_imag()[0]) for point in solve(polynomial_expr.diff(x_sym), x_sym)]

        # Find the domain range for the plot
        if critical_real_points and x_intercepts:
            domain_min, domain_max = min(min(critical_real_points), min(x_intercepts)), max(max(critical_real_points), max(x_intercepts))
            delta = round((abs(domain_min)+ abs(domain_max))/max(abs(domain_min), abs(domain_max)),2)
            domain_min = domain_min - delta if domain_min == 0 else domain_min + copysign(delta, domain_min)
            domain_max += copysign(delta, domain_max)
        elif critical_real_points:
            domain_min, domain_max = min(critical_real_points), max(critical_real_points)
        elif x_intercepts and not (len(x_intercepts) == 1 and x_intercepts[0] == 0):
            domain_min, domain_max = min(x_intercepts), max(x_intercepts)
        else:
            domain_min, domain_max = -10.0, 10.0
            

        # Precalculate the function values at domain_min and domain_max
        poly_func_domain_min = poly_func(domain_min)
        poly_func_domain_max = poly_func(domain_max)

        # Find the maximum absolute value among domain_min, domain_max, and the function values at those points
        axis_lims = np.max(np.abs([domain_min, domain_max, poly_func_domain_min, poly_func_domain_max]))

        # Generate x and y values for the plot
        xtemp = np.linspace(-axis_lims, axis_lims, 200)

        # Calculate y values over the entire domain range
        y = poly_func(xtemp)
        x = xtemp[np.abs(y) < axis_lims]
        y = y[np.abs(y) < axis_lims]
        
        # Check behavior at the ends of the x range
        while abs(poly_func(x[0])) < axis_lims:
            x = np.insert(x, 0, x[0] - (x[1] - x[0]))
            y = np.insert(y, 0, poly_func(x[0]))
            

        while abs(poly_func(x[-1])) < axis_lims:
            x = np.append(x, x[-1] + (x[-1] - x[-2]))
            y = np.append(y, poly_func(x[-1]))

        axis_lims = max(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1]))



        # Create the plot
        fig, ax = plt.subplots(dpi=100)
        ax.plot(x, y, color='black', linewidth=0.4)

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticks([-axis_lims, 0, axis_lims])
        ax.set_yticks([-axis_lims, 0, axis_lims])

        ax.set_xlim(-axis_lims, axis_lims)
        
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        # plt.show()

        degree_dir = os.path.join(base_dir, f'degree_{degree}')

        # Create the degree directory if it doesn't exist
        if not os.path.exists(degree_dir):
            os.makedirs(degree_dir)

        # Save the figure with corresponding label as the file name
        label_str = ','.join(map(str, coeffs_rounded))  # Save the actual coefficients as a string
        image_name = f'polynomial_degree_{degree}_{i}.png'
        image_path = os.path.join(degree_dir, image_name)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        with open(csv_file_path, mode='a', newline='') as label_file:
            label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if not csv_exists:
                label_writer.writerow(['image_path', 'coeffs', 'dr_ratio'])
                csv_exists = True

            label_str = ','.join(map(str, coeffs_rounded))
            dr_ratio_str = f"{round((abs(y[0]) + abs(y[-1])) / (abs(x[0]) +abs(x[-1])), 2)}"
            label_writer.writerow([image_path, label_str, dr_ratio_str])

print(f"Dataset and labels saved in {base_dir}")

# Verification step
for degree in range(1, 5):
    degree_dir = os.path.join(base_dir, f'degree_{degree}')
    file_count = len(os.listdir(degree_dir))
    print(f"Folder {degree_dir} contains {file_count} images.")
