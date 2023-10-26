import os
import pdb; 
import csv
import numpy as np
from sympy import *
from sympy.plotting import plot
import matplotlib.pyplot as plt
from math import copysign
import random
from itertools import chain

# Generate dataset for polynomials

num_samples = 20  # Set the number of samples to x for each degree

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

degree_range = 5

# Customizes the appearance of a sympy plot within a matplotlib subplot.
def customize_sympyplot_appearance(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['left'].set_position('zero')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['top'].set_color('none')
    backend.ax.set_xlabel('')
    backend.ax.set_ylabel('')
    backend.ax.set_xticks([])
    backend.ax.set_yticks([])
    backend.ax.axis('off')
    plt.close(backend.fig)
    


for degree in range(1, degree_range):
    samples_generated = 0
    for i in range(num_samples):
        while samples_generated < num_samples:

            # Generate coefficients
            # Ensure the leading zeros for coefficients of lower degrees
            coeffs = [round(Float(random.uniform(-coeff_range, coeff_range)),2 ) for _ in range(degree)] + [round(Float(random.uniform(-coeff_range, coeff_range)), 2)]
            for zeros in range (4-degree):
                coeffs.insert(0, Float(0))

            # Define the symbolic variable and the polynomial expression
            x_sym = symbols('x', real=True)
            polynomial_expr = sum([coeff * x_sym ** (4 - j) for j, coeff in enumerate(coeffs[:-1])] + [coeffs[-1]])
            
            # Find intercepts, critical points, and other values for plotting
            x_intercepts = [root for root in solveset(polynomial_expr, x_sym) if root.is_real]
            x_intercepts_inverse = [-root for root in x_intercepts]
            combined_x_intercepts = x_intercepts_inverse + x_intercepts
            y_intercepts = [polynomial_expr.subs(x_sym, 0)] if isinstance(polynomial_expr.subs(x_sym, 0), list) else [polynomial_expr.subs(x_sym, 0)]
            critical_real_points = [point for point in solveset(polynomial_expr.diff(x_sym), x_sym) if point.is_real]
            xvals = combined_x_intercepts + critical_real_points
            yvals = [polynomial_expr.subs(x_sym, i) for i in xvals] + y_intercepts
            
            
            # Calculate axis limits and values for the graph
            abs_values = list(np.concatenate([np.absolute(xvals), np.abs(yvals)]))
            axis_lims = max(abs_values) + (0.2*max(abs_values) )
            xlims = list(chain(*[solve(Eq(polynomial_expr, -axis_lims), x_sym), solve(Eq(polynomial_expr, axis_lims), x_sym)]))
            xmin, xmax = min(xlims), max(xlims)
            ymin, ymax = polynomial_expr.subs(x_sym, xmin), polynomial_expr.subs(x_sym, xmax)           
            
            # Calculate the ratio of y-values to x-values
            dr_ratio = round(( (abs(max(ymax, ymin)) +  abs(min(ymax,ymin)))/(abs(max(xmax, xmin)) + abs(min(xmax,xmin)))), 2)
            if dr_ratio >= 100: 
                continue
            samples_generated+=1
            axis_lims = max(abs(ymax), abs(ymin), abs(xmin), abs(xmax))
            
           
            # Plot the polynomial function
            func_plot = plot(polynomial_expr,(x_sym, -axis_lims, axis_lims), xlim=[-axis_lims, axis_lims], ylim=[-axis_lims, axis_lims], show=False, line_color = 'black')
            
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig, ax = plt.subplots(figsize=(50*px, 50*px))
            customize_sympyplot_appearance(func_plot, ax)
            
            
    
            # Create the degree directory if it doesn't exist
            degree_dir = os.path.join(base_dir, f'degree_{degree}')
            if not os.path.exists(degree_dir):
                os.makedirs(degree_dir)
                
            # Save the actual coefficients as a string
            label_str = ','.join(map(str, coeffs))  # Save the actual coefficients as a string
            image_path = os.path.join(degree_dir, f"polynomial_degree_{degree}_{samples_generated}.png")
            
            # Save the data to a CSV file
        
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            
            with open(csv_file_path, mode='a', newline='') as label_file:
                    label_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    if not csv_exists:
                        label_writer.writerow(['image_path', 'coeffs', 'dr_ratio'])
                        csv_exists = True

                    label_str = ','.join(map(str, coeffs))
                    dr_ratio_str = f"{dr_ratio}"
                    label_writer.writerow([image_path, label_str, dr_ratio_str])
            


print(f"Dataset and labels saved in {base_dir}")

# Verification step
for degree in range(1, 5):
    degree_dir = os.path.join(base_dir, f'degree_{degree}')
    file_count = len(os.listdir(degree_dir))
    print(f"Folder {degree_dir} contains {file_count} images.")
