import os
import rasterio

def create_tfw_from_geotiff(image_path, output_dir):
    """
    Generates a .tfw world file by extracting georeferencing
    information from a GeoTIFF or similar raster file.

    Args:
        image_path (str): The full path to the georeferenced image.
        output_dir (str): The directory where the .tfw file will be saved.
    """
    try:
        with rasterio.open(image_path) as src:
            if src.transform.is_identity:
                print(f"--> SKIPPING: The image '{os.path.basename(image_path)}' is not georeferenced.")
                return

            transform = src.transform

            c_center = transform.c + (transform.a / 2.0)
            f_center = transform.f + (transform.e / 2.0)

            tfw_content = (
                f"{transform.a}\n"
                f"{transform.d}\n"
                f"{transform.b}\n"
                f"{transform.e}\n"
                f"{c_center}\n"
                f"{f_center}\n"
            )

        base, ext = os.path.basename(image_path).rsplit('.', 1)
        if ext.lower() in ('tif', 'tiff'):
            tfw_ext = 'tfw'
        elif ext.lower() in ('jpg', 'jpeg'):
            tfw_ext = 'jgw'
        elif ext.lower() == 'png':
            tfw_ext = 'pgw'
        else:
            tfw_ext = 'wld'

        tfw_filename = f"{base}.{tfw_ext}"
        tfw_path = os.path.join(output_dir, tfw_filename)

        with open(tfw_path, 'w') as f:
            f.write(tfw_content)

        print(f"--> SUCCESS: Created world file for '{os.path.basename(image_path)}'")

    except rasterio.errors.RasterioIOError as e:
        print(f"--> ERROR: Could not read '{os.path.basename(image_path)}'. Not a valid raster file. Details: {e}")
    except Exception as e:
        print(f"--> ERROR: An unexpected error occurred with '{os.path.basename(image_path)}': {e}")

def process_single_file():
    """Handles the logic for processing a single image file."""
    image_path = input("Enter the full path to the georeferenced image: ")
    output_dir = input("Enter the directory to save the world file: ")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
    elif not os.path.isdir(output_dir):
        print(f"Error: Output directory not found at '{output_dir}'")
    else:
        create_tfw_from_geotiff(image_path, output_dir)

def process_batch_folder():
    """Handles the logic for processing a whole folder of TIFF images."""
    input_dir = input("Enter the path to the folder containing your TIFF images: ")
    output_dir = input("Enter the path to the folder where world files should be saved: ")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory not found at '{output_dir}'")
        return

    print(f"\nScanning folder: {input_dir}")
    file_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            file_count += 1
            image_path = os.path.join(input_dir, filename)
            create_tfw_from_geotiff(image_path, output_dir)

    if file_count == 0:
        print("No TIFF files (.tif, .tiff) were found in the specified folder.")
    else:
        print(f"\nBatch processing complete. Processed {file_count} files.")

if __name__ == '__main__':
    while True:
        print("\n--- World File Generator ---")
        print("Please choose a processing mode:")
        print("  1: Process a single image file")
        print("  2: Process a batch of TIFFs in a folder")
        print("  Q: Quit")

        choice = input("Enter your choice (1, 2, or Q): ").strip().lower()

        if choice == '1':
            process_single_file()
            break
        elif choice == '2':
            process_batch_folder()
            break
        elif choice == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter '1', '2', or 'Q'.")
