import os
import subprocess
import shutil

def check_gdal():
    """Checks if gdal_translate is available in the system's PATH."""
    if shutil.which("gdal_translate") is None:
        print("---")
        print("ERROR: gdal_translate command not found.")
        print("This script requires the GDAL command-line tools to be installed and in your system's PATH.")
        print("Please visit https://gdal.org/download.html for installation instructions.")
        print("---")
        return False
    return True

def create_tfw_with_gdal(image_path, output_dir):
    """
    Generates a world file using the gdal_translate command by creating and
    then renaming the world file associated with a temporary raster.

    Args:
        image_path (str): The full path to the georeferenced image.
        output_dir (str): The directory where the world file will be saved.
    """
    base_orig, _ = os.path.splitext(os.path.basename(image_path))
    temp_base = f"{base_orig}_temp_output"
    temp_raster_path = os.path.join(output_dir, f"{temp_base}.tif")

    command = [
        "gdal_translate",
        "-co", "TFW=YES",
        image_path,
        temp_raster_path
    ]

    try:
        # Execute the command, hide output unless there's an error
        subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

        # Find the world file GDAL created (e.g., ..._temp_output.tfw)
        generated_world_file = None
        for filename in os.listdir(output_dir):
            if filename.startswith(temp_base) and filename.lower().endswith('w'):
                generated_world_file = filename
                break

        if generated_world_file:
            # Rename the world file to match the original image name
            _, world_ext = os.path.splitext(generated_world_file)
            final_world_filename = f"{base_orig}{world_ext}"

            source_path = os.path.join(output_dir, generated_world_file)
            dest_path = os.path.join(output_dir, final_world_filename)

            # Overwrite if a file with the same name already exists
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(source_path, dest_path)

            print(f"--> SUCCESS: Created '{final_world_filename}'")
        else:
            # This can happen if the source image is not georeferenced.
            print(f"--> SKIPPING: No world file generated for '{os.path.basename(image_path)}'. The image may not be georeferenced.")

    except FileNotFoundError:
        # This case is handled by check_gdal(), but is here as a fallback.
        print(f"--> ERROR: Could not find the 'gdal_translate' command.")
    except subprocess.CalledProcessError as e:
        # This error means gdal_translate ran but returned an error code.
        print(f"--> ERROR: GDAL failed to process '{os.path.basename(image_path)}'.")
        print(f"   GDAL message: {e.stderr.strip()}")
    except Exception as e:
        print(f"--> ERROR: An unexpected error occurred with '{os.path.basename(image_path)}': {e}")
    finally:
        # The temporary raster file is intentionally not deleted as per user request.
        pass

def process_single_file():
    """Handles the logic for processing a single image file."""
    image_path = input("Enter the full path to the georeferenced image: ")
    output_dir = input("Enter the directory to save the world file: ")

    if not os.path.isfile(image_path):
        print(f"\nError: Image file not found at '{image_path}'")
    elif not os.path.isdir(output_dir):
        print(f"\nError: Output directory not found at '{output_dir}'")
    else:
        create_tfw_with_gdal(image_path, output_dir)

def process_batch_folder():
    """Handles the logic for processing a whole folder of georeferenced images."""
    input_dir = input("Enter the path to the folder containing your images: ")
    output_dir = input("Enter the path to the folder where world files should be saved: ")

    if not os.path.isdir(input_dir):
        print(f"\nError: Input directory not found at '{input_dir}'")
        return
    if not os.path.isdir(output_dir):
        print(f"\nError: Output directory not found at '{output_dir}'")
        return

    supported_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.img')
    print(f"\nScanning folder: {input_dir}")
    file_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            file_count += 1
            image_path = os.path.join(input_dir, filename)
            create_tfw_with_gdal(image_path, output_dir)

    if file_count == 0:
        print(f"No supported image files {supported_extensions} were found in the specified folder.")
    else:
        print(f"\nBatch processing complete. Processed {file_count} files.")

if __name__ == '__main__':
    if check_gdal():
        while True:
            print("\n--- World File Generator (using GDAL) ---")
            print("Please choose a processing mode:")
            print("  1: Process a single image file")
            print("  2: Process a batch of images in a folder")
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
                print("\nInvalid choice. Please enter '1', '2', or 'Q'.")
