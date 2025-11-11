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
            # Check if the image has a valid geotransform
            if src.transform.is_identity:
                print(f"Error: The image '{image_path}' is not georeferenced.")
                return

            transform = src.transform

            # The world file format is:
            # Line 1: A: pixel size in the x-direction
            # Line 2: D: row rotation (typically 0)
            # Line 3: B: column rotation (typically 0)
            # Line 4: E: pixel size in the y-direction (typically negative)
            # The world file requires the coordinates of the CENTER of the top-left pixel.
            # rasterio's transform gives the coordinate of the TOP-LEFT CORNER.
            # We must adjust for this by adding half a pixel's width to C and half a pixel's height to F.
            c_center = transform.c + (transform.a / 2.0)
            f_center = transform.f + (transform.e / 2.0)

            # The world file format is:
            # Line 1: A: pixel size in the x-direction
            # Line 2: D: row rotation (typically 0)
            # Line 3: B: column rotation (typically 0)
            # Line 4: E: pixel size in the y-direction (typically negative)
            # Line 5: C: x-coordinate of the center of the upper left pixel
            # Line 6: F: y-coordinate of the center of the upper left pixel
            tfw_content = (
                f"{transform.a}\n"
                f"{transform.d}\n"
                f"{transform.b}\n"
                f"{transform.e}\n"
                f"{c_center}\n"
                f"{f_center}\n"
            )

        # Determine the .tfw file path
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

        print(f"Successfully created world file: {tfw_path}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error: Could not read the image file. Ensure the path is correct and the file is a valid raster. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    image_path = input("Enter the full path to the georeferenced image: ")
    output_dir = input("Enter the directory to save the world file: ")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
    elif not os.path.isdir(output_dir):
        print(f"Error: Output directory not found at '{output_dir}'")
    else:
        create_tfw_from_geotiff(image_path, output_dir)
