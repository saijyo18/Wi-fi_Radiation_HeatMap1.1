import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, zoom

class Colors:
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"


# Function to create heatmap
def create_heatmap(csv_path, image_path, output_dir, value_column, x_col='x', y_col='y',vminimum = 0,vmaximum = 30):
    # Load the data into a pandas data frame
    data = pd.read_csv(csv_path)

    # Create a pivot table from the dataframe data
    raw_heatmap_data = data.pivot_table(index=y_col, columns=x_col, values=value_column)

    # Fill missing values and upscale the data grid
    raw_heatmap_data = raw_heatmap_data.fillna(0)
    up_sample_factor = 5  # Factor to increase resolution
    smoothed_data = zoom(raw_heatmap_data, up_sample_factor)

    # Apply Gaussian smoothing to the up-sampled data
    smoothed_data = gaussian_filter(smoothed_data, sigma=1.5)

    # Create and save the heatmap with a color bar
    plt.figure(figsize=(10, 10))
    sns.heatmap(smoothed_data, cmap="Spectral", vmin=vminimum, vmax=vmaximum, cbar=True, xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.tight_layout()
    heatmap_cb_path = f"{output_dir}/heatmap_cb_{value_column}.png"
    plt.savefig(heatmap_cb_path, bbox_inches='tight', pad_inches=0, dpi=96)
    plt.close()

    # Create and save the heatmap without a color bar
    plt.figure(figsize=(10, 10))
    sns.heatmap(smoothed_data, cmap="Spectral", cbar=False, vmin=vminimum, vmax=vmaximum, xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.tight_layout()
    heatmap_path = f"{output_dir}/heatmap_{value_column}.png"
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0, dpi=96)
    plt.close()

    # Load the saved heatmap and the map image
    map_image = cv2.imread(image_path)
    heatmap = cv2.imread(heatmap_path)

    # Resize the heatmap to match the map dimensions
    heatmap = cv2.resize(heatmap, (map_image.shape[1], map_image.shape[0]))

    # Overlay the heatmap onto the map
    blended_image = cv2.addWeighted(map_image, 0.3, heatmap, 0.5, 0)

    # Save the overlaid heatmap
    output_path = f"{output_dir}/final_heatmap_{value_column}.png"
    cv2.imwrite(output_path, blended_image)
    print(f"{Colors.GREEN}Overlay heatmap saved to {output_path}{Colors.RESET}")

# Main script
csv_file = r"C:\SNU\PhyProject\PhysicsProjectData.csv"  # Path to CSV file
floor_map = r"C:\SNU\PhyProject\map.jpeg"  # Path to map image
output_directory = r"C:\SNU\PhyProject\heatmaps"  # Output directory for the heatmaps

# List of columns to generate heatmaps for
columns_to_visualize = ['InternetSpeedClean', 'InternetSpeedNoise', 'UploadSpeed']

# Loop through each column and generate the heatmap
for column in columns_to_visualize:
    print(f"{Colors.YELLOW}Generating heatmap for {column}...{Colors.RESET}")
    create_heatmap(csv_file, floor_map, output_directory, column, "xcoordinate", "ycoordinate")

# Generate heatmap for RSSI with custom parameters
print(f"\n{Colors.YELLOW}Generating heatmap for RSSI...{Colors.RESET}")
create_heatmap(csv_file,floor_map,output_directory,    "RSSI","xcoordinate",    "ycoordinate",vminimum= -100, vmaximum= -60)

