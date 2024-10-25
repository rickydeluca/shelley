import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the folder containing CSV files and the output folder
input_folder = 'results/'
output_folder = 'plots/'
os.makedirs(output_folder, exist_ok=True)

# Function to load all CSV files from the input folder
def load_csv_files(folder):
    csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]
    dataframes = [pd.read_csv(os.path.join(folder, file)) for file in csv_files]
    return pd.concat(dataframes, ignore_index=True)

# Load all data
df = load_csv_files(input_folder)

# Get unique train_ratios and noise conditions
train_ratios = df['train_ratio'].unique()
noise_types = [('noise_add', 'noise_rm')]

# Assign colors to each algorithm (model)
models = df['model'].unique()
colors = plt.cm.get_cmap('tab10', len(models)).colors  # Generate colors for each algorithm
model_colors = {model: colors[i] for i, model in enumerate(models)}

# Plot function
def plot_performance(data, noise_type, train_ratio, output_folder):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        model_data = data[data['model'] == model]
        
        # Plot the dots for accuracy
        ax.scatter(model_data[noise_type], model_data['avg_acc'], label=model, color=model_colors[model])
        
        # Connect the dots with a dotted line
        ax.plot(model_data[noise_type], model_data['avg_acc'], linestyle='--', color=model_colors[model])
        
        # Plot shaded area for standard deviation
        ax.fill_between(model_data[noise_type], 
                        model_data['avg_acc'] - model_data['std_acc'], 
                        model_data['avg_acc'] + model_data['std_acc'], 
                        color=model_colors[model], alpha=0.2)
    
    # Set labels and title
    noise_name = "Noise (removal)" if noise_type == 'noise_rm' else 'Noise (addition)'
    ax.set_xlabel(noise_name)
    ax.set_ylabel('Average Accuracy')
    ax.set_title(f'Train Ratio: {train_ratio} | {noise_name}')
    ax.legend()
    
    # Save the plot using tight layout
    plot_filename = f'plot_{noise_type}_train_ratio_{train_ratio}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close()

# Iterate over each train_ratio and noise condition to generate plots
for train_ratio in train_ratios:
    for noise_add, noise_rm in noise_types:
        # Filter data based on train_ratio and noise type
        data_add = df[(df['train_ratio'] == train_ratio) & (df['noise_rm'] == 0.0)]
        data_rm = df[(df['train_ratio'] == train_ratio) & (df['noise_add'] == 0.0)]
        
        # Generate and save plots for both noise types
        plot_performance(data_add, 'noise_add', train_ratio, output_folder)
        plot_performance(data_rm, 'noise_rm', train_ratio, output_folder)

print(f"Plots saved in: {output_folder}")