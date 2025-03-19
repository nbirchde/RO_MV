import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import time
from datetime import datetime
import os

def evaluate_sequence(sequence, processing_times, priorities, deadlines):
    """Evaluate a job sequence to get makespan and total weighted tardiness."""
    num_jobs = len(sequence)
    num_machines = len(processing_times[0])
    completion_times = np.zeros((num_jobs, num_machines))
    
    # First job, first machine
    first_job = sequence[0]
    completion_times[0][0] = processing_times[first_job][0]
    
    # First job on all machines
    for m in range(1, num_machines):
        completion_times[0][m] = completion_times[0][m-1] + processing_times[first_job][m]
    
    # All other jobs
    for j in range(1, num_jobs):
        job = sequence[j]
        # First machine depends only on previous job
        completion_times[j][0] = completion_times[j-1][0] + processing_times[job][0]
        
        # Other machines
        for m in range(1, num_machines):
            completion_times[j][m] = max(completion_times[j][m-1], completion_times[j-1][m]) + processing_times[job][m]
    
    # Calculate makespan (completion time of last job on last machine)
    makespan = completion_times[-1][-1]
    
    # Calculate total weighted tardiness
    total_tardiness = 0
    for j in range(num_jobs):
        job = sequence[j]
        completion_time = completion_times[j][-1]
        tardiness = max(0, completion_time - deadlines[job])
        total_tardiness += tardiness * priorities[job]
    
    return makespan, total_tardiness

def load_instance(file_path):
    """Load problem instance from CSV file."""
    processing_times = []
    priorities = []
    deadlines = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            values = [float(x) for x in row]
            priorities.append(values[0])
            deadlines.append(values[1])
            processing_times.append(values[2:])
    
    return np.array(processing_times), np.array(priorities), np.array(deadlines)

def plot_pareto_front(solutions_file, instance_file, output_file=None):
    """Plot the Pareto front from the solutions file."""
    # Load problem instance
    print(f"Loading problem instance from {instance_file}")
    processing_times, priorities, deadlines = load_instance(instance_file)
    
    # Load and evaluate solutions
    makespans = []
    tardiness = []
    
    print(f"Evaluating solutions from {solutions_file}")
    with open(solutions_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            sequence = [int(x) for x in row]
            makespan, tard = evaluate_sequence(sequence, processing_times, priorities, deadlines)
            makespans.append(makespan)
            tardiness.append(tard)
    
    # Create the plot with a timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with larger points and more opacity
    plt.scatter(makespans, tardiness, c='blue', s=100, alpha=0.8, label='Solutions')
    
    # Connect points with lines to show front
    # Sort points by makespan for correct line drawing
    sorted_indices = np.argsort(makespans)
    sorted_makespans = [makespans[i] for i in sorted_indices]
    sorted_tardiness = [tardiness[i] for i in sorted_indices]
    plt.plot(sorted_makespans, sorted_tardiness, 'b--', alpha=0.5, linewidth=2)
    
    # Add point numbers for easier reference
    for i, (x, y) in enumerate(zip(makespans, tardiness)):
        plt.annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Labels and title with timestamp
    plt.xlabel('Makespan', fontsize=14)
    plt.ylabel('Total Weighted Tardiness', fontsize=14)
    plt.title(f'Pareto Front for Flowshop Problem\nGenerated: {timestamp}', fontsize=16)
    
    # Grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some statistics to the plot
    stats_text = f'Number of solutions: {len(makespans)}\n'
    stats_text += f'Makespan range: [{min(makespans):.1f}, {max(makespans):.1f}]\n'
    stats_text += f'Tardiness range: [{min(tardiness):.1f}, {max(tardiness):.1f}]'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Better style
    plt.tight_layout()
    
    # Force axes to refresh (avoids caching issues)
    plt.gcf().canvas.draw()
    
    # Generate timestamp-based filenames
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output paths
    if output_file:
        output_path = output_file
    else:
        script_dir = Path(__file__).parent
        output_path = script_dir / "pareto_front.png"
    
    # Always create a timestamped version
    timestamped_path = output_path.parent / f"pareto_front_{timestamp_file}.png"
    
    # Save both the standard and timestamped versions
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(timestamped_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: \n  {output_path} \n  {timestamped_path}")
    
    # Open the image file automatically (works on macOS)
    try:
        os.system(f"open {timestamped_path}")
    except:
        print(f"Could not automatically open the image. Please find it at: {timestamped_path}")

if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Define paths relative to script location
    solutions_file = script_dir / "birch_delacalle_nicholas.csv"
    instance_file = script_dir / "instance.csv"
    
    # Create the plot with auto-generated filename
    plot_pareto_front(solutions_file, instance_file)