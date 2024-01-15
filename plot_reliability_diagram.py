import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Function to plot the reliability diagram
def plot_reliability_diagram(csv_file_path):
    # Read the CSV file with semicolon separator
    df = pd.read_csv(csv_file_path, sep=';')
    
    # Calculate the accuracy for each bin
    df['accuracy'] = df['absolute_occurrences'] / df['total_samples_in_bin']
    # Replace NaN values with 0 (which occur if the total_samples_in_bin is 0)
    df['accuracy'].fillna(0, inplace=True)
    
    # Create the plot
    fig, ax = plt.subplots()
    # Plot the accuracy for each bin as a bar chart
    ax.bar(df['predicted_probability'], df['accuracy'], width=0.1, color='black', label='Accuracy')
    # Plot the gap (difference between predicted probability and actual frequency) as a bar chart
    ax.bar(df['predicted_probability'], df['predicted_probability'] - df['actual_frequency'],
           width=0.1, bottom=df['actual_frequency'], color='salmon', alpha=0.5, label='Gap')
    # Plot the perfectly calibrated line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Set the plot labels and title
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram - ' + os.path.basename(csv_file_path))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='best')
    
    # Save the plot as an image file
    output_file_path = os.path.splitext(csv_file_path)[0] + '_reliability_diagram.png'
    plt.savefig(output_file_path)
    plt.close()
    print(f'Reliability diagram saved as {output_file_path}')

# Directory containing your CSV files
csv_directory = '/path/to/your/csv/directory'

# Find all CSV files in the directory
csv_files = list(Path(csv_directory).glob('*.csv'))

# Loop through each file and plot the reliability diagram
for file_path in csv_files:
    plot_reliability_diagram(str(file_path))
    print(f'Processed file: {file_path}')