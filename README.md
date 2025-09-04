# Geo-Segmenter

Geo-Segmenter is a powerful, interactive tool for segmenting objects in geospatial imagery. It combines a feature-rich web interface with the cutting-edge "Segment Anything Model" (SAM) from Meta AI, allowing for precise, prompt-based object extraction from local orthophotos, satellite imagery via Google Earth Engine, or even 3D map snapshots.

<img width="1907" height="920" alt="image" src="https://github.com/user-attachments/assets/730c7917-89ff-4a55-86b4-be8659cdc6c2" />


## Key Features

- **Multi-Source Image Loading**:
    - **Local Files**: Load your own georeferenced orthophotos (e.g., `.tif`, `.jpg`) using an accompanying world file (`.tfw`, `.jgw`).
    - **Google Earth Engine**: Fetch and display satellite imagery directly from Google Earth Engine by specifying an Image ID, region of interest, bands, and date range.
    - **3D Map Snapshot**: Navigate a 3D Google Map view, capture any scene, and use it for segmentation.
- **Advanced Segmentation**:
    - **AI-Powered**: Utilizes the Segment Anything Model (SAM) for high-quality, zero-shot object segmentation.
    - **Multiple Prompts**: Guide the model by using positive/negative point clicks or by drawing a bounding box.
    - **Manual Digitization**: Draw polygons manually for full control.
- **Geospatial Awareness**:
    - **Georeferencing**: Properly handles georeferenced imagery and can transform segmented pixels back into geographic coordinates.
    - **Manual Georeferencing**: For unreferenced snapshots, you can place Ground Control Points (GCPs) on the 3D map before capture to establish a coordinate system.
- **Comprehensive Layer & Data Management**:
    - **Layered Display**: Manage multiple vector layers, controlling their visibility, opacity, and draw order.
    - **Import/Export**:
        - Import external vector data from GeoJSON or zipped Shapefiles.
        - Export your segmented polygons to GeoJSON or Shapefile format.
    - **Stash and Load**: Save your entire session (polygons and view context) to a "stash" and reload it later.
- **Interactive User Interface**:
    - **Rich UI**: Built with Tailwind CSS for a modern and responsive experience.
    - **Canvas Tools**: Pan, zoom, and interact directly with the image and vector data.
    - **Symbology Editor**: Customize the visual style (fill color, stroke, etc.) of your vector layers.
    - **Dark/Light Mode**: Switch between themes for user comfort.
    - **CPU/GPU Control**: Choose whether the backend runs the AI model on the CPU or GPU.

## How It Works

Geo-Segmenter consists of two main components:

1.  **Frontend (`Geo-Segmenter.html`)**: A standalone HTML file containing a sophisticated vanilla JavaScript application. It provides the user interface for loading data, drawing on the canvas, creating prompts, and managing layers. It communicates with the backend via HTTP requests.
2.  **Backend (`app.py`)**: A Python server built with Flask. It handles the heavy lifting:
    - Serves the image data to the frontend.
    - Loads and manages the SAM model on the specified device (CPU or GPU).
    - Receives segmentation prompts (points, boxes) from the frontend.
    - Runs the AI model to predict object masks.
    - Performs georeferencing calculations.
    - Handles file import/export operations (Shapefile, GeoJSON).

## Setup and Installation

Follow these steps to get Geo-Segmenter running on your local machine.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up a Python Environment

It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

### 4. Download the Model Checkpoint

The application requires the Segment Anything Model (SAM) checkpoint file.

- **Model**: ViT-B (vit_b)
- **Filename**: `sam_vit_b_01ec64.pth`

Download the file from the official source:
[**ViT-B SAM Model Checkpoint**](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Place the downloaded `sam_vit_b_01ec64.pth` file in the root directory of the project, alongside `app.py`.

### 5. (Optional) Authenticate Google Earth Engine

To use the Google Earth Engine integration, you need to authenticate your machine. Run the following command in your terminal and follow the prompts:

```bash
earthengine authenticate
```

## Usage

1.  **Start the Backend Server**:
    Run the `app.py` script from your terminal. Make sure your Python virtual environment is activated.

    ```bash
    python app.py
    ```
    The server will start, load the SAM model, and listen for requests on `http://127.0.0.1:5000`.

2.  **Open the Frontend**:
    In your web browser (Chrome or Firefox recommended), open the `Geo-Segmenter.html` file directly.
    - You can typically do this by double-clicking the file or using `File > Open File...` in your browser.

3.  **Use the Application**:
    - The application should connect to the backend automatically (the status indicator will turn green).
    - Select an image source (Local Files, GEE, or 3D Map) and provide the required inputs.
    - Once an image is loaded, use the segmentation tools (points, box) to start extracting features.
    - Manage your results in the Layers panel on the right.

## Dependencies

- **Backend**:
    - `torch` & `torchvision`
    - `opencv-python`
    - `numpy`
    - `Flask` & `Flask-CORS`
    - `segment-anything` (Meta AI's official package)
    - `Pillow`
    - `rasterio`
    - `earthengine-api`
    - `pyshp`

- **Frontend**:
    - Tailwind CSS (via CDN)
    - `html2canvas` (for map snapshots)

## Author & Copyright

Copyright © 2024 K.Tanaval
