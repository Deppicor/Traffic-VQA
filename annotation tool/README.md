# Annotation Software Guide

## I. Introduction

This graphical annotation tool, developed using PyQt5, facilitates the annotation of **aligned multimodal image pairs**, specifically optical (OPT) and thermal (TIR) images. It allows users to annotate various aspects such as image matching, weather conditions, scene types, and traffic-related cognitive elements. Annotations are automatically saved as JSON files to a user-specified directory. This tool is designed for building datasets for tasks like Multimodal Visual Question Answering (VQA) and image understanding.

**Note on Modality:**
*   Currently, this tool is optimized for annotating **paired optical and thermal images**.
*   Support for single-modality annotation (e.g., only optical images) is planned for a future release.

## II. System Requirements

*   **Python:** Version 3.6 or higher.
*   **Required Libraries:**
    ```bash
    pip install numpy opencv-python PyQt5
    ```

## III. Getting Started

1.  Ensure `annotation_tool.py` (or your script name) is in your project's root directory.
2.  Run the tool from your terminal:
    ```bash
    python annotation_tool.py
    ```

## IV. User Guide

### 1. Interface Overview

The main interface is organized as follows:

*   **Top-Left (Folder Selection):**
    *   `Browse OPT`: Select the folder containing optical images.
    *   `Browse TIR` (TIR): Select the folder containing thermal images.
    *   `Browse SAVE`: Select the directory where annotation files will be saved.
*   **Middle-Left (Image Display):**
    *   Displays the selected optical and thermal images side-by-side.
    *   Shows the current image name and annotation progress above the images.
*   **Bottom-Left (Annotation Log):**
    *   Lists saved attribute annotations for the current image, allowing for review.
*   **Right (Annotation Options - Scrollable Panel):**
    *   Lists primary annotation categories (e.g., Match & Weather, Scene Theme, Distance/Location, etc.).
    *   Selecting a category reveals sub-options for detailed attribute annotation.

    **Primary Annotation Categories:** Match & Weather, Scene Theme, Distance/Location (LocDis), Contain/Presence (PresContain), Deduce (Custom Attributes), Traffic Elements, Residential Elements, Agricultural Features, Industrial Features, UAV Parameters.

### 2. Initial Setup: Selecting Folders

Before starting, you **must** select the three required folders:

1.  Click `Browse OPT` to choose your optical image folder.
2.  Click `Browse TIR` to choose your thermal image folder.
3.  Click `Browse SAVE` to choose your annotation save directory.

Once selected, the tool will automatically attempt to load and align image pairs, displaying the first pair and progress information.

### 3. General Annotation Workflow

For most categories, the general workflow is:

1.  **Select a Primary Category** (e.g., "Traffic Elements") from the checkbox in the right-hand panel.
2.  **Select Sub-Categories/Objects** using the dropdowns that appear for that category.
3.  **Fill in Attributes** (e.g., Number, Location, Shape) using the corresponding dropdowns.
4.  Click the **`Confirm ... Item`** or **`Confirm ... Pair`** button specific to that category to add the current set of attributes to a temporary list for that image.
5.  Repeat steps 2-4 for other objects/elements within the *same primary category* for the current image.
6.  Click the **`Finalize ...`** (or `Input End` in older versions) button specific to that category to bundle all "Confirmed" items under that category and store them as part of the current image's overall attributes.
7.  Repeat steps 1-6 for other primary categories you wish to annotate for the current image.

### 4. Detailed Annotation Categories

*(The detailed descriptions for categories 4.1 to 4.10 remain largely the same as your original, focusing on the *purpose* and *sub-options*. The workflow is covered by the "General Annotation Workflow" above. Key interactive elements like "Measure Distance/Area" are highlighted below.)*

#### 4.1. Match & Weather (Image-level Attributes)

*   **Purpose:** Annotate overall image characteristics and perceived weather.
*   **Sub-options:** Image Matching (alignment), Fog conditions, Low Light conditions.

#### 4.2. Scene Theme (Image-level Attributes)

*   **Purpose:** Define the general scene environment.
*   **Sub-options:** Area Type (Residential/n-Residential), Scene Macro Category (Urban/Rural).

#### 4.3. Distance/Location (LocDis)

*   **Purpose:** Describe spatial relationships between two objects (Object A and Object B).
*   **Workflow:**
    1.  Define Object A (position, cluster, category).
    2.  Define Object B (position, cluster, category).
    3.  Define their Relationship (distance, orientation).
    4.  Click `Confirm D/L Pair`.
    5.  **Optional - Measure Distance:**
        *   Click `Measure Distance`. An OpenCV window will pop up displaying the **optical image**.
        *   Click two points to define a line. The pixel distance will be displayed in the OpenCV window.
        *   **Note on Resolution:** By default, 1 pixel = 1 meter. See section "V. Important Notes > Distance and Area Measurement" for customization.
    6.  After all pairs are confirmed, click `Finalize D/L`.

#### 4.4. Contain/Presence (PresContain)

*   **Purpose:** Annotate the presence and attributes of specific land cover classes.
*   **Workflow:**
    1.  Select LandCover Class, then Subset Class.
    2.  Select Attributes (Number, Location, Shape, Area, Length, Distribution, Image Quality).
    3.  **Optional - Measure Area:**
        *   Click `Measure Area`. An OpenCV window will pop up displaying the **optical image**.
        *   Click multiple points clockwise to define a polygon. The pixel area will be displayed.
        *   **Note on Resolution:** By default, 1 pixel = 1 meter². See section "V. Important Notes > Distance and Area Measurement" for customization.
    4.  Click `Confirm C/P Item`.
    5.  After all items for this LandCover/Subset are confirmed, or for all PresContain items, click `Finalize C/P`.
    6.  **To Modify/Delete:** Select the LandCover class from the dropdown and click `Delete Current C/P LandCover` to remove all annotations for that specific LandCover for the current image.

#### 4.5. Deduce (Custom Attributes)

*   **Purpose:** Add custom key-value attribute pairs if predefined categories are insufficient.
*   **Workflow:**
    1.  Check the "Deduce" option.
    2.  Enter your custom attribute key (e.g., "Atmospheric_Haze_Level") in the first text box.
    3.  Enter the corresponding value (e.g., "Medium") in the second text box.
    4.  Click `Submit Attr 1`.
    5.  Repeat for a second custom attribute if needed using the other pair of text boxes and `Submit Attr 2`.

#### 4.6. Traffic Elements

*   **Purpose:** Annotate various traffic-related elements and behaviors.
*   *(Similar workflow to PresContain: Select Category, Object, Attributes, Confirm Item, Finalize Traffic, Delete Current Traffic LandCover)*

#### 4.7. Residential Elements

*   **Purpose:** Annotate aspects specific to residential areas.
*   *(Similar workflow to PresContain: Select Category, Object, Attributes, Confirm Item, Finalize Residential, Delete Current Residential LandCover)*

#### 4.8. Agricultural Features

*   **Purpose:** Annotate basic agricultural features.
*   **Sub-options:** Agricultural roads present, Agricultural water bodies present.

#### 4.9. Industrial Features

*   **Purpose:** Annotate basic industrial site characteristics.
*   **Sub-options:** Industrial facilities present, Industrial scale, Industrial location.

#### 4.10. UAV Parameters

*   **Purpose:** Annotate UAV capture parameters.
*   **Sub-options:** Drone shooting height, Drone shooting angle.

### 5. Navigation and Saving

*   **`Previous`:** Saves current image's attributes to memory and loads the previous image pair.
*   **`Next`:** Saves current image's attributes to memory and loads the next image pair.
*   **`Save Current Image Attrs`:** Saves all attributes defined for the **current image** to an in-memory dictionary. This is useful for explicitly saving before making major changes or if you are not ready to move to the next image. This action is also implicitly done when navigating with `Next` or `Previous`.
*   **`Save All to File`:** Writes all in-memory attributes (for all images processed and "saved to memory" so far in the session) to the `image_attributes.json` file in the designated "SAVE" directory. **This is the crucial step to persist your work to disk.**

## V. Important Notes

*   **File Naming and Pairing:**
    *   Ensure optical (OPT) and thermal (TIR) image filenames correspond one-to-one for correct pairing. For example, `image001_rgb.jpg` in the OPT folder and `image001_tir.jpg` in the TIR folder. The tool attempts to match based on common prefixes if extensions or suffixes like `_rgb`/`_tir` differ.
*   **Skipping Annotations:** You can choose to skip any category or sub-option if it is not relevant to the current image by simply not checking its main checkbox or not selecting options from its dropdowns.
*   **Annotation File:** A single JSON file named `image_attributes.json` will be created in your selected "SAVE" directory. This file stores a dictionary where keys are image filenames and values are dictionaries of their annotated attributes.
*   **Distance and Area Measurement (Resolution):**
    *   The `Measure Distance` and `Measure Area` features operate on the **optical image**.
    *   By default, the tool assumes a resolution where **1 pixel equals 1 meter** for distance and **1 pixel equals 1 square meter** for area.
    *   If your imagery has a different Ground Sampling Distance (GSD) or resolution (e.g., 0.5 meters/pixel), you need to adjust the calculation in the Python script.
    *   To do this, locate the `ImageMeasurement` class in `annotation_tool.py`.
        *   In the `_distance_mouse_event` method, find the line where `distance` is calculated (e.g., `distance = math.sqrt(...)`). Modify it to `distance = math.sqrt(...) / your_resolution_factor`.
        *   In the `_area_mouse_event` method, find where `area_val` is calculated (e.g., `area_val = abs(polygon_area(polygon))`). Modify it to `area_val = abs(polygon_area(polygon)) / (your_resolution_factor ** 2)`.
        *   For example, if 1 pixel = 0.5 meters, `your_resolution_factor` would be `1 / 0.5 = 2`. So, you would divide pixel distance by 2, and pixel area by 2². (Alternatively, multiply by GSD for distance, and GSD² for area).
*   **Extending Annotation Categories:**
    *   The tool is designed with some extensibility in mind for adding more detailed sub-categories.
    *   Many dropdown options are populated from Python dictionaries within the `AnnotationWindow` class (e.g., `self.TR2ObjList`, `self.RA2ObjList`, `self.LC2SubList`).
    *   **Quick Method to Add Sub-options:**
        1.  Open `annotation_tool.py` in a text editor.
        2.  Use `Ctrl+F` (or `Cmd+F`) to search for the parent category or a related existing sub-option. For example, to add "Trampling the lawn" under a "pedestrian violation" type, you might search for `"pedestrian traffic violation"` within the `self.TR2ObjList` dictionary.
        3.  Locate the list associated with that key (e.g., the list of strings for "pedestrian traffic violation").
        4.  Add your new sub-option (e.g., `"Trampling the lawn"`) to that list.
        5.  Save the `annotation_tool.py` file and restart the annotation tool. Your new option should appear in the dropdown.
    *   This primarily applies to adding more choices to existing dropdowns. Adding entirely new *primary* categories or complex interactive elements would require more significant code changes.

## VI. Contact and Feedback

For suggestions, feature requests, or bug reports, please post issues or contact the development team via wa2124207@stu.shu.edu.cn.
