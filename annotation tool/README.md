# Annotation Software Guide

## I. Introduction

This graphical annotation tool, developed using PyQt5, facilitates the annotation of aligned optical (OPT) and thermal (TIR) images. It allows users to annotate various aspects such as image matching, weather conditions, scene types, and traffic-related cognitive elements. Annotations are automatically saved to a user-specified directory. This tool is designed for building datasets for tasks like Multimodal Visual Question Answering (VQA) and image understanding.

## II. System Requirements

*   **Python:** Version 3.6 or higher.
*   **Required Libraries:**
    ```bash
    pip install numpy opencv-python PyQt5
    ```

## III. Getting Started

1.  Ensure `annotation_tool.py` is in your project's root directory.
2.  Run the tool from your terminal:
    ```bash
    python annotation_tool.py
    ```

## IV. User Guide

### 1. Interface Overview

The main interface is organized as follows:

*   **Top-Left (Folder Selection):**
    *   `Browse OPT`: Select the folder containing optical images.
    *   `Browse TIR`: Select the folder containing thermal images.
    *   `Browse SAVE`: Select the directory where annotation files will be saved.
*   **Middle-Left (Image Display):**
    *   Displays the selected optical and thermal images side-by-side.
    *   Shows the current image name and annotation progress above the images.
*   **Bottom-Left (Annotation Log):**
    *   Lists saved annotations for the current image, allowing for review.
*   **Right (Annotation Options):**
    *   Lists primary annotation categories (e.g., Match, Theme, Distance/Location, etc.).
    *   Selecting a category reveals sub-options for detailed annotation.

    **Primary Annotation Categories:** Match, Theme, Distance/Location (LocDis), Contain/Presence (PresContain), Deduce, Traffic, Residential, Agricultural, Industrial, UAV.

### 2. Initial Setup: Selecting Folders

Before starting, you **must** select the three required folders:

1.  Click `Browse OPT` to choose your optical image folder.
2.  Click `Browse THE` to choose your thermal image folder.
3.  Click `Browse SAVE` to choose your annotation save directory.

Once selected, the tool will automatically load and align image pairs, displaying the first pair and progress information.

### 3. General Annotation Workflow

For most categories, the general workflow is:

1.  **Select a Primary Category** (e.g., "Traffic") from the right-hand panel.
2.  **Select Sub-Categories/Objects** from the options that appear.
3.  **Fill in Attributes** (e.g., Number, Location, Shape).
4.  Click **`Confirm`** to save the annotation for the current object/element.
5.  Repeat steps 2-4 for other objects within the same primary category.
6.  Click **`Input End`** (if available for the category) to finalize annotations for that category for the current image. This usually bundles all "Confirmed" items under that category.

### 4. Detailed Annotation Categories

#### 4.1. Match (Image-level Attributes)

*   **Purpose:** Annotate overall image characteristics.
*   **Sub-options:**
    *   **Image Matching:** `almost match`, `partial match`, `not match`.
    *   **Weather Simulation:**
        *   Fog: `mist`, `not mist`, `not sure`.
        *   Low Light: `dark`, `not dark`, `not sure`.

#### 4.2. Theme (Image-level Attributes)

*   **Purpose:** Define the general scene environment.
*   **Sub-options:**
    *   **Area Type:** `Residential` or `n-Residential` (non-Residential).
    *   **Scene Macro Category:** `Urban` or `Rural`.

#### 4.3. Distance/Location (LocDis)

*   **Purpose:** Describe spatial relationships between two objects (Object A and Object B).
*   **Workflow:**
    1.  **Object A:**
        *   Select location.
        *   Specify if area: `none` or `cluster`.
        *   Select category (e.g., `woodland`, `wide road`, `low-rise residential building`, etc. - *see tool for full list*).
    2.  **Object B:** Repeat the same selections as for Object A.
    3.  **Relationship:**
        *   Specify distance relationship.
        *   Specify orientation of B relative to A (e.g., `above`, `below`, `left`, etc.).
    4.  Click `Confirm` to save the A-B relationship.
    5.  **Optional - Measure Distance:**
        *   Click `Length and Distance`.
        *   In the popped-up optical image, click two points to measure the closest distance. The system displays the calculated length.
    6.  After annotating all A-B pairs for the current image, click `Input End` to summarize LocDis information.

#### 4.4. Contain/Presence (PresContain)

*   **Purpose:** Annotate the presence and attributes of specific land cover classes.
*   **Workflow:**
    1.  **Select LandCover Class:** (e.g., `agricultural area`, `building`, `road`, `sports field` - *see tool for full list*).
    2.  **Select Subset Class:** (e.g., if LandCover is `building`, Subset could be `low-rise residential building` - *see tool for specific mappings*).
    3.  **Select Attributes:**
        *   `Number`, `Location`, `Shape` (e.g., `Straight`, `Curved`, `Rotundity`), `Area`, `Length`, `Distribution` (e.g., `Clustered`, `Isolated`), `Better` (image quality: `optical`, `thermal`, `almost same`).
    4.  **Optional - Measure Area:**
        *   Click `Measure Area`.
        *   In the popped-up optical image, draw a polygon clockwise around the target to get its area.
    5.  Click `Confirm` to save the object's annotation. Repeat for other objects.
    6.  Click `Input End` to summarize all PresContain data for the current image.
    7.  **To Modify:** Select a marked object and click `Delete Current LandCover` to remove it.

#### 4.5. Deduce

*   **Purpose:** Create question-answer pairs for reasoning tasks.
*   **Workflow:**
    1.  Check the "Deduce" option.
    2.  **First Question-Answer Pair:**
        *   Enter question one in "Please enter your question one".
        *   Enter a **single-word answer** in "Please enter the answer".
        *   Click `submit first`.
    3.  **Second Question-Answer Pair:**
        *   Enter question two in "Please enter your question two".
        *   Enter a **single-word answer** in "Please enter the answer".
        *   Click `submit second`.
    *   Both Q&A pairs will be saved as reasoning data.

#### 4.6. Traffic

*   **Purpose:** Annotate various traffic-related elements and behaviors.
*   **Workflow:**
    1.  **Select LandCover Class:** (e.g., `road type`, `vehicle`, `pedestrian traffic violation`, `vehicle behavior` - *see tool for extensive list*).
    2.  **Select Subset Class:** (e.g., if LandCover is `vehicle`, Subset could be `car`; if `vehicle traffic violation`, Subset could be `illegal parking` - *see tool for specific mappings*).
    3.  **Select Attributes:** `Number`, `Location`, `Quality` (image quality: `optical`, `thermal`, `almost same`).
    4.  Click `Confirm` to save. Repeat for other traffic elements.
    5.  Click `Input End` to summarize all Traffic data.
    6.  **To Modify:** Select a marked object and click `Delete Current Traffic LandCover`.

#### 4.7. Residential

*   **Purpose:** Annotate aspects specific to residential areas.
*   **Workflow:**
    1.  **Select LandCover Class:** `living environment` or `construction type`.
    2.  **Select Object:**
        *   If `living environment`: `recreational area`, `river`, `linear walkway`, etc.
        *   If `construction type`: `low-rise residential buildings`, `high-rise non-residential buildings`, etc.
    3.  **Select Attributes:** `Number`, `Location`, `Quality` (image quality: `optical`, `thermal`, `almost same`).
    4.  Click `Confirm` to save. Repeat for other residential elements.
    5.  Click `Input End` to summarize.
    6.  **To Modify:** Select a marked object and click `Delete Current Residential LandCover`.

#### 4.8. Agricultural

*   **Purpose:** Annotate basic agricultural features.
*   **Sub-options:**
    *   Agricultural roads present: `yes`, `no`.
    *   Agricultural water bodies present: `yes`, `no`.

#### 4.9. Industrial

*   **Purpose:** Annotate basic industrial site characteristics.
*   **Sub-options:**
    *   Industrial facilities present: `yes`, `no`.
    *   Industrial scale: `small`, `medium`, `large`.
    *   Industrial location (within image frame): `top`, `bottom`, `left`, `right`, `center`, etc.

#### 4.10. UAV

*   **Purpose:** Annotate UAV capture parameters.
*   **Sub-options:**
    *   Drone shooting height (meters): `150-250`, `250-400`, `400-550`, `none`.
    *   Drone shooting angle: `vertical`, `oblique`.

### 5. Navigation and Submission

*   **`Pre`:** Go to the previous image pair.
*   **`Next`:** Go to the next image pair.
*   **`Submit`:** Saves all annotations made for the **current image** to memory (does not yet write to file). This action is implicitly done when navigating with `Next` or `Pre` as well.
*   **`Generate`:** Writes all "submitted" annotations (for all images processed so far in the session) to their respective annotation files in the designated "SAVE" directory.

## V. Important Notes

*   **File Naming:** Ensure optical (OPT) and thermal (TIR) image filenames match one-to-one for correct pairing (e.g., `image001.jpg` in OPT folder and `image001.jpg` in TIR folder).
*   **Skipping Annotations:** You can choose to skip certain categories or sub-options if they are not relevant to the current image.
*   **Annotation Files:** Annotation files are automatically named based on the corresponding image name (e.g., `image001.json` or `image001.txt`, depending on save format).

## VI. Contact and Feedback

For suggestions, feature requests, or bug reports, please contact the development team via [wa2124207@stu.ahu.edu.cn].
