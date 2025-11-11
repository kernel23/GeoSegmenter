import argparse
import rasterio
from rasterio.transform import from_bounds

def create_tfw(image_path, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    """
    Generates a .tfw world file for a given orthomosaic image.

    Args:
        image_path (str): The full path to the orthomosaic image.
        top_left_x (float): The x-coordinate of the top-left corner.
        top_left_y (float): The y-coordinate of the top-left corner.
        bottom_right_x (float): The x-coordinate of the bottom-right corner.
        bottom_right_y (float): The y-coordinate of the bottom-right corner.
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
        base, ext = image_path.rsplit('.', 1)
        if ext.lower() == 'tif' or ext.lower() == 'tiff':
            tfw_ext = 'tfw'
        elif ext.lower() == 'jpg' or ext.lower() == 'jpeg':
            tfw_ext = 'jgw'
        elif ext.lower() == 'png':
            tfw_ext = 'pgw'
        else:
            tfw_ext = 'wld'

        tfw_path = f"{base}.{tfw_ext}"

        with open(tfw_path, 'w') as f:
            f.write(tfw_content)

        print(f"Successfully created world file: {tfw_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a .tfw world file for an orthomosaic image.')
    parser.add_argument('image_path', type=str, help='Path to the orthomosaic image file.')
    parser.add_argument('top_left_x', type=float, help='X-coordinate of the top-left corner.')
    parser.add_argument('top_left_y', type=float, help='Y-coordinate of the top-left corner.')
    parser.add_argument('bottom_right_x', type=float, help='X-coordinate of the bottom-right corner.')
    parser.add_argument('bottom_right_y', type=float, help='Y-coordinate of the bottom-right corner.')

    args = parser.parse_args()

    create_tfw(args.image_path, args.top_left_x, args.top_left_y, args.bottom_right_x, args.bottom_right_y)
