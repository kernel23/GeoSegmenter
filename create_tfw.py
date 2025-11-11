import os
import rasterio
from rasterio.transform import from_bounds

def create_tfw(image_path, top_left_x, top_left_y, bottom_right_x, bottom_right_y, output_dir):
    """
    Generates a .tfw world file for a given orthomosaic image.

    Args:
        image_path (str): The full path to the orthomosaic image.
        top_left_x (float): The x-coordinate of the top-left corner.
        top_left_y (float): The y-coordinate of the top-left corner.
        bottom_right_x (float): The x-coordinate of the bottom-right corner.
        bottom_right_y (float): The y-coordinate of the bottom-right corner.
        output_dir (str): The directory where the .tfw file will be saved.
    """
    try:
        with rasterio.open(image_path) as src:
            width = src.width
            height = src.height

        transform = from_bounds(top_left_x, bottom_right_y, bottom_right_x, top_left_y, width, height)

        # Build the .tfw file content
        tfw_content = (
            f"{transform.a}\n"
            f"{transform.d}\n"
            f"{transform.b}\n"
            f"{transform.e}\n"
            f"{transform.c}\n"
            f"{transform.f}\n"
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

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    image_path = input("Enter the full path to the orthomosaic image: ")
    top_left_x = float(input("Enter the X-coordinate of the top-left corner: "))
    top_left_y = float(input("Enter the Y-coordinate of the top-left corner: "))
    bottom_right_x = float(input("Enter the X-coordinate of the bottom-right corner: "))
    bottom_right_y = float(input("Enter the Y-coordinate of the bottom-right corner: "))
    output_dir = input("Enter the directory to save the .tfw file: ")

    create_tfw(image_path, top_left_x, top_left_y, bottom_right_x, bottom_right_y, output_dir)
