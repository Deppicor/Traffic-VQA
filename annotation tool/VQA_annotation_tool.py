# -*- coding: utf-8 -*-
# This tool is designed for annotating optical and thermal infrared images.

import json
import sys
import os
import random
import math
import numpy as np
import cv2 # OpenCV for image processing and display for distance/area measurement
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QCheckBox,
    QFileDialog,
    QLineEdit,
    QComboBox,
    QTextEdit,
    QScrollArea, # Added for scrollable annotation panel
    QVBoxLayout, # Added for layout within scroll area
)

# --- Helper Functions for Geometric Calculations ---
def coss_multi(v1, v2):
    """Calculates the 2D cross product of two vectors."""
    return v1[0] * v2[1] - v1[1] * v2[0]

def polygon_area(polygon):
    """
    Calculates the area of a polygon given its vertices.
    Args:
        polygon (np.array): A NumPy array of shape (n, 2) representing n vertices.
    Returns:
        float: The area of the polygon.
    """
    n = len(polygon)
    if n < 3:
        return 0 # A polygon needs at least 3 vertices
    vectors = np.zeros((n, 2))
    for i in range(n):
        vectors[i, :] = polygon[i, :] - polygon[0, :] # Vectors relative to the first vertex
    area = 0
    for i in range(1, n):
        area += coss_multi(vectors[i - 1, :], vectors[i, :]) / 2
    return area

# --- Image Measurement Class (using OpenCV) ---
class ImageMeasurement:
    """
    Handles interactive distance and area measurement on an image using OpenCV.
    """
    def __init__(self, image_path):
        self.path = image_path
        self.img_original = cv2.imread(self.path) # Load the original image
        self.img_display = self.img_original.copy() # Create a copy for drawing
        self.coordinates_distance = [] # Stores points for distance measurement
        self.coordinates_area = []     # Stores points for area measurement (polygon vertices)
        self.distance_count = 0
        self.area_count = 0

    def _distance_mouse_event(self, event, x, y, flags, param):
        """Mouse callback function for distance measurement."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates_distance.append([x, y])
            cv2.circle(self.img_display, (x, y), 3, (0, 255, 0), thickness=-1) # Draw a point
            if len(self.coordinates_distance) % 2 == 0 and len(self.coordinates_distance) >= 2:
                # Draw line between the last two points
                p1 = tuple(self.coordinates_distance[-2])
                p2 = tuple(self.coordinates_distance[-1])
                cv2.line(self.img_display, p1, p2, (255, 0, 0), 2)
            cv2.imshow("Measure Distance", self.img_display)

        elif event == cv2.EVENT_MBUTTONDOWN: # Middle mouse button to calculate and display distance
            if len(self.coordinates_distance) >= 2:
                self.distance_count += 1
                p1 = self.coordinates_distance[-2]
                p2 = self.coordinates_distance[-1]
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                cv2.putText(
                    self.img_display,
                    f"Distance_{self.distance_count}: {distance:.2f}",
                    (10, 30 + self.distance_count * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    thickness=2,
                )
                cv2.imshow("Measure Distance", self.img_display)
                # self.coordinates_distance = [] # Optional: Clear points after measurement

    def measure_distance(self):
        """Opens an OpenCV window to interactively measure distances."""
        self.img_display = self.img_original.copy() # Reset display image
        self.coordinates_distance = []
        cv2.namedWindow("Measure Distance", cv2.WINDOW_AUTOSIZE) # Or WINDOW_NORMAL for resizable
        cv2.setMouseCallback("Measure Distance", self._distance_mouse_event)
        cv2.imshow("Measure Distance", self.img_display)
        cv2.waitKey(0) # Wait until a key is pressed
        cv2.destroyWindow("Measure Distance")

    def _area_mouse_event(self, event, x, y, flags, param):
        """Mouse callback function for area measurement."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates_area.append([x, y])
            cv2.circle(self.img_display, (x, y), 3, (0, 255, 0), thickness=-1)
            if len(self.coordinates_area) > 1:
                # Draw line to the previous point to form the polygon
                cv2.line(self.img_display, tuple(self.coordinates_area[-2]), tuple(self.coordinates_area[-1]), (255, 0, 0), 2)
            cv2.imshow("Measure Area", self.img_display)

        elif event == cv2.EVENT_MBUTTONDOWN: # Middle mouse button to calculate and display area
            if len(self.coordinates_area) >= 3: # Need at least 3 points for an area
                self.area_count += 1
                polygon = np.array(self.coordinates_area)
                area_val = abs(polygon_area(polygon)) # Use abs for positive area
                # Draw the completed polygon (optional, can make it messy)
                # cv2.polylines(self.img_display, [polygon.astype(np.int32)], isClosed=True, color=(0,0,255), thickness=2)
                cv2.putText(
                    self.img_display,
                    f"Area_{self.area_count}: {area_val:.2f}",
                    (10, 30 + self.area_count * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    thickness=2,
                )
                cv2.imshow("Measure Area", self.img_display)
                self.coordinates_area = [] # Clear points for the next area measurement

    def measure_area(self):
        """Opens an OpenCV window to interactively measure areas (polygons)."""
        self.img_display = self.img_original.copy() # Reset display image
        self.coordinates_area = []
        cv2.namedWindow("Measure Area", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Measure Area", self._area_mouse_event)
        cv2.imshow("Measure Area", self.img_display)
        cv2.waitKey(0)
        cv2.destroyWindow("Measure Area")

# --- Utility Function to Get Image Paths ---
def get_img_paths(directory, extensions=(".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
    """
    Retrieves all image file paths and names from a given directory.
    Args:
        directory (str): The path to the folder containing images.
        extensions (tuple): A tuple of valid image file extensions.
    Returns:
        tuple: (list of full image paths, list of image filenames)
    """
    img_paths = []
    img_names = []
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found: {directory}")
        return img_paths, img_names

    for filename in sorted(os.listdir(directory)): # Sort to ensure consistent order
        if filename.lower().endswith(extensions):
            img_paths.append(os.path.join(directory, filename))
            img_names.append(filename)
    return img_paths, img_names

# --- Main Annotation Window Class ---
class AnnotationWindow(QWidget):
    def __init__(self):
        super().__init__()

        # --- Paths and Image Data ---
        self.selected_folder_rgb = ""       # Path to the OPT (optical/RGB) image folder
        self.selected_folder_tir = ""       # Path to the THE (thermal infrared) image folder
        self.save_folder = ""               # Path to the folder where annotation files will be saved

        self.img_paths_rgb = []             # List of full paths to RGB images
        self.img_paths_tir = []             # List of full paths to TIR images
        self.rgb_names = []                 # List of RGB image filenames
        self.tir_names = []                 # List of TIR image filenames

        self.num_rgb_images = 0             # Total number of RGB images
        self.num_tir_images = 0             # Total number of TIR images
        self.current_image_index = 0        # Index of the currently displayed image pair

        # --- Annotation Data Storage ---
        # self.annotation_dict = {}         # REMOVED: Was for storing Q&A pairs.
        self.attribution_dict = {}          # Stores image attributes: {img_name: {attribute1: value, ...}, ...}
                                            # Also stores a "counter" key for resuming annotation.

        self.current_image_attributes = {}  # Attributes for the currently displayed image: {attribute1: value, ...}
        self.current_rgb_name = ""          # Filename of the current RGB image
        self.current_tir_name = ""          # Filename of the current TIR image

        # --- UI State and Helper Variables ---
        self.allow_modification = True      # Flag to enable/disable modification of annotation UI elements (prevents loops)
        self.current_landcover_for_delete = "None" # Tracks the LandCover class for targeted deletion in PresContain
        self.current_traffic_landcover_for_delete = "None"  # For Traffic category
        self.current_residential_landcover_for_delete = "None" # For Residential category


        # --- UI Dimensions and Titles ---
        self.setWindowTitle("Optical-Thermal Image Annotation Tool")
        self.setGeometry(200, 100, 1580, 770) # left, top, width, height
        self.img_panel_width = 450
        self.img_panel_height = 450

        # --- Data Structures for Annotation Options (Dropdowns, etc.) ---
        # These dictionaries map primary categories to their sub-categories/objects.
        # Used to dynamically populate QComboBoxes.
        self.RA2ObjList = { # Residential Area to Object List
            "living environment": ["Object", "recreational area", "commercial area", "construction area", "river", "lake", "linear walkway", "curved walkway", "no visible walkway"],
            "construction type": ["Object", "low-rise residential building", "high-rise residential building", "low-rise non-residential building", "high-rise non-residential building"],
            "LandCover Class": ["Object Class"], # Default/placeholder
        }
        self.TR2ObjList = { # Traffic to Object List
            "road type": ["Object", "main city road", "street", "quick road", "residential street", "alley", "intersection", "lane merge", "pedestrian crossing", "bridge", "non-motorized road", "unpaved road", "bus lane", "gridline", "overhead walkway", "other paved road"],
            "vehical": ["Object", "car", "large vehicle", "other vehicle"],
            "pedestrian": ["Object", "single pedestrian", "pedestrian group"],
            "road facility": ["Object", "motor vehicle parking spot", "non-motorised parking spot", "lane marking", "road divider"],
            "road condition": ["Object", "normal pavement", "damaged pavement", "road construction"],
            "vehicle traffic violation": ["Object", "illegal parking", "go against one-way traffic", "Illegal lane change", "run the red light", "vehicle on solid line"],
            "non-motor vehicle violation": ["Object", "illegal passenger carrying", "wrong-way driving", "running red light", "improper lane usage", "improper parking", "no safety helmet"],
            "pedestrian traffic violation": ["Object", "failure to use crosswalks", "walking on non-sidewalks", "run the red light", "other violations"],
            "vehicle behavior": ["Object", "lane change", "vehicle turn", "vehicle U-turn", "overtake", "vehicle queuing", "traffic congestion", "too close to another car"],
            "non-motor vehicle behavior": ["Object", "waiting at traffic light", "normal driving in non-motor lane"],
            "pedestrian behavior": ["Object", "wait for a traffic light", "walk on the crosswalk", "walk on the sidewalk"],
            "abnormal traffic situation": ["Object", "traffic accident", "traffic jam"],
            "traffic participant interaction": ["Object", "vehicle yielding to pedestrian", "vehicle waiting for boarding", "vehicle waiting for alighting", "bus temporary stop", "vehicle entering parking lot", "vehicle exiting parking lot"],
            "LandCover Class": ["Object Class"], # Default/placeholder
        }
        self.LC2SubList = { # LandCover to Subset List (for PresContain)
            "building": ["Subset", "low-rise residential building", "high-rise residential building", "low-rise non-residential building", "high-rise non-residential building"],
            "vegetation area": ["Subset", "woodland", "grassland", "other vegetation area"],
            "water area": ["Subset", "ditch", "pond", "river", "sea", "lake", "other water area"],
            "road": ["Subset", "wide road", "narrow road"],
            "agricultural area": ["Subset", "agricultural area"],
            "wasteland": ["Subset", "wasteland"],
            "intersection": ["Subset", "intersection"],
            "parking area": ["Subset", "parking area"],
            "park": ["Subset", "park"],
            "concrete floor": ["Subset", "concrete floor"],
            "sports field": ["Subset", "basketball court", "baseball field", "football field", "tennis courts", "athletic track"],
            "pier": ["Subset", "pier"],
            "beach": ["Subset", "beach"],
            "airport": ["Subset", "airport"], # Although 'airport' is a LandCover, it can also be a subset if a larger area is 'airport'
            "apron": ["Subset", "apron"],
            "LandCover Class": ["Subset Class"], # Default/placeholder
        }

        # --- Temporary Storage for Complex Annotations (per image) ---
        self.dis_loc_details = {}  # Stores distance/location pairs: {(A,B): [distance_index, location_text, distance_text]}
        self.contain_details = {}  # Stores PresContain details: {landcover_class: [[subset_idx, subset_text], [attr1_idx, attr1_text], ...]}
        self.traffic_details = {}  # Stores Traffic details: {landcover_class: [[object_idx, object_text], [attr1_idx, attr1_text], ...]}
        self.residential_details = {} # Stores Residential details: {landcover_class: [[object_idx, object_text], [attr1_idx, attr1_text], ...]}

        # --- Common Lists for Dropdowns ---
        self.location_options = ["above", "below", "left", "right", "upper left", "upper right", "bottom left", "bottom right"]
        # self.object_dict=['dog','cat','plane','house','twon'] # REMOVED: Example, not used directly if dynamic

        # --- UI Element Definitions ---
        # Folder Selection UI
        self.headline_folder_rgb = QLabel("1. Select OPT (Optical) Image Folder:", self)
        self.headline_folder_tir = QLabel("2. Select THE (Thermal) Image Folder:", self)
        self.headline_folder_save = QLabel("3. Select Annotation Save Folder:", self)
        self.selected_folder_label_rgb = QLabel(self) # Displays selected RGB folder path
        self.selected_folder_label_tir = QLabel(self) # Displays selected TIR folder path
        self.selected_folder_label_save = QLabel(self) # Displays selected save folder path
        self.browse_button_rgb = QtWidgets.QPushButton("Browse OPT", self)
        self.browse_button_tir = QtWidgets.QPushButton("Browse THE", self)
        self.browse_button_save = QtWidgets.QPushButton("Browse SAVE", self)

        # Image Display UI
        self.image_box_rgb = QLabel(self) # Displays RGB image
        self.image_box_tir = QLabel(self) # Displays TIR image
        self.rgb_name_label = QLabel(self)  # Displays current RGB image name
        self.progress_bar_rgb = QLabel(self) # Displays RGB image progress (e.g., "1 of 100")
        self.tir_name_label = QLabel(self) # Displays current TIR image name
        self.progress_bar_tir = QLabel(self) # Displays TIR image progress

        # Annotation Log/Display UI
        self.display_anno_log = QTextEdit(self) # Displays a log of confirmed annotations for the current image
        self.display_anno_log.setReadOnly(True)

        # --- Annotation Categories (Checkboxes) ---
        self.question_headline = QLabel("Select annotation categories and provide details:", self)
        self.chk_match = QCheckBox("Match & Weather", self)
        self.chk_theme = QCheckBox("Scene Theme", self)
        self.chk_dis_loc = QCheckBox("Distance/Location", self)
        self.chk_contain = QCheckBox("Contain/Presence", self)
        self.chk_deduce = QCheckBox("Deduce (Custom Q&A)", self) # Retained this for flexibility if needed later for non-VQA attributes
        self.chk_traffic = QCheckBox("Traffic Elements", self)
        self.chk_residential = QCheckBox("Residential Elements", self)
        self.chk_agricultural = QCheckBox("Agricultural Features", self)
        self.chk_industrial = QCheckBox("Industrial Features", self)
        self.chk_uav = QCheckBox("UAV Parameters", self)

        # --- Annotation Detail Widgets (ComboBoxes, LineEdits, Buttons per category) ---
        # These will be initialized in self.init_ui() and added to a scrollable layout.

        # Match & Weather
        self.match_options_box = QComboBox(self)
        self.mist_options_box = QComboBox(self)
        self.night_options_box = QComboBox(self)

        # Theme
        self.theme_residential_box = QComboBox(self)
        self.theme_urban_rural_box = QComboBox(self)

        # Distance/Location
        self.disloc_pos_a_box = QComboBox(self)
        self.disloc_obj_a_box = QComboBox(self)
        self.disloc_cluster_a_box = QComboBox(self)
        self.disloc_pos_b_box = QComboBox(self)
        self.disloc_obj_b_box = QComboBox(self)
        self.disloc_cluster_b_box = QComboBox(self)
        self.disloc_distance_box = QComboBox(self)
        self.disloc_relation_box = QComboBox(self)
        self.btn_submit_disloc = QtWidgets.QPushButton("Confirm D/L Pair", self)
        self.btn_measure_distance = QtWidgets.QPushButton("Measure Distance", self)
        self.btn_finish_disloc = QtWidgets.QPushButton("Finalize D/L", self)

        # Contain/Presence
        self.contain_landcover_box = QComboBox(self)
        self.contain_subset_box = QComboBox(self)
        self.contain_number_box = QComboBox(self)
        self.contain_location_box = QComboBox(self)
        self.contain_shape_box = QComboBox(self)
        self.contain_area_box = QComboBox(self)
        self.contain_length_box = QComboBox(self)
        self.contain_distribution_box = QComboBox(self)
        self.contain_quality_box = QComboBox(self) # "Better" renamed to "Quality"
        self.btn_submit_contain = QtWidgets.QPushButton("Confirm C/P Item", self)
        self.btn_measure_area = QtWidgets.QPushButton("Measure Area", self)
        self.btn_finish_contain = QtWidgets.QPushButton("Finalize C/P", self)
        self.btn_delete_contain_landcover = QtWidgets.QPushButton("Delete Current C/P LandCover", self)

        # Deduce (Retained for attribute input, not VQA pair generation)
        self.deduce_q1_input = QLineEdit("Enter custom attribute 1 key", self) # Changed placeholder
        self.deduce_a1_input = QLineEdit("Enter attribute 1 value", self)      # Changed placeholder
        self.deduce_q2_input = QLineEdit("Enter custom attribute 2 key", self) # Changed placeholder
        self.deduce_a2_input = QLineEdit("Enter attribute 2 value", self)      # Changed placeholder
        self.btn_submit_deduce1 = QtWidgets.QPushButton("Submit Attr 1", self)
        self.btn_submit_deduce2 = QtWidgets.QPushButton("Submit Attr 2", self)

        # Traffic
        self.traffic_landcover_box = QComboBox(self)
        self.traffic_object_box = QComboBox(self)
        self.traffic_number_box = QComboBox(self)
        self.traffic_location_box = QComboBox(self)
        self.traffic_quality_box = QComboBox(self)
        self.btn_submit_traffic = QtWidgets.QPushButton("Confirm Traffic Item", self)
        self.btn_finish_traffic = QtWidgets.QPushButton("Finalize Traffic", self)
        self.btn_delete_traffic_landcover = QtWidgets.QPushButton("Delete Current Traffic LandCover", self)

        # Residential
        self.residential_landcover_box = QComboBox(self)
        self.residential_object_box = QComboBox(self)
        self.residential_number_box = QComboBox(self)
        self.residential_location_box = QComboBox(self)
        self.residential_quality_box = QComboBox(self)
        self.btn_submit_residential = QtWidgets.QPushButton("Confirm Residential Item", self)
        self.btn_finish_residential = QtWidgets.QPushButton("Finalize Residential", self)
        self.btn_delete_residential_landcover = QtWidgets.QPushButton("Delete Current Residential LandCover", self)

        # Agricultural
        self.agri_road_box = QComboBox(self)
        self.agri_water_box = QComboBox(self)

        # Industrial
        self.ind_facility_box = QComboBox(self)
        self.ind_scale_box = QComboBox(self)
        self.ind_location_box = QComboBox(self)

        # UAV
        self.uav_height_box = QComboBox(self)
        self.uav_angle_box = QComboBox(self)

        # --- Navigation and Control Buttons ---
        self.btn_previous_image = QtWidgets.QPushButton("Previous", self)
        self.btn_next_image = QtWidgets.QPushButton("Next", self)
        self.btn_submit_all_current = QtWidgets.QPushButton("Save Current Image Attrs", self) # Renamed for clarity
        self.btn_generate_file = QtWidgets.QPushButton("Save All to File", self) # Renamed for clarity

        # --- Lists for easier management of UI elements (used in init_pannel) ---
        self.all_checkboxes = [
            self.chk_match, self.chk_theme, self.chk_dis_loc, self.chk_contain,
            self.chk_deduce, self.chk_traffic, self.chk_residential,
            self.chk_agricultural, self.chk_industrial, self.chk_uav
        ]
        self.all_comboboxes_and_lineedits = [ # For resetting them
            self.match_options_box, self.mist_options_box, self.night_options_box,
            self.theme_residential_box, self.theme_urban_rural_box,
            self.disloc_pos_a_box, self.disloc_obj_a_box, self.disloc_cluster_a_box,
            self.disloc_pos_b_box, self.disloc_obj_b_box, self.disloc_cluster_b_box,
            self.disloc_distance_box, self.disloc_relation_box,
            self.contain_landcover_box, self.contain_subset_box, self.contain_number_box,
            self.contain_location_box, self.contain_shape_box, self.contain_area_box,
            self.contain_length_box, self.contain_distribution_box, self.contain_quality_box,
            self.deduce_q1_input, self.deduce_a1_input, self.deduce_q2_input, self.deduce_a2_input,
            self.traffic_landcover_box, self.traffic_object_box, self.traffic_number_box,
            self.traffic_location_box, self.traffic_quality_box,
            self.residential_landcover_box, self.residential_object_box, self.residential_number_box,
            self.residential_location_box, self.residential_quality_box,
            self.agri_road_box, self.agri_water_box,
            self.ind_facility_box, self.ind_scale_box, self.ind_location_box,
            self.uav_height_box, self.uav_angle_box
        ]
        # Lists of QComboBoxes for specific categories (used for populating from stored attributes)
        self.contain_attribute_widgets = [ # Order matters: subset, number, location, shape, area, length, distribution, quality
            self.contain_subset_box, self.contain_number_box, self.contain_location_box,
            self.contain_shape_box, self.contain_area_box, self.contain_length_box,
            self.contain_distribution_box, self.contain_quality_box
        ]
        self.traffic_attribute_widgets = [ # Order: object, number, location, quality
            self.traffic_object_box, self.traffic_number_box, self.traffic_location_box, self.traffic_quality_box
        ]
        self.residential_attribute_widgets = [ # Order: object, number, location, quality
            self.residential_object_box, self.residential_number_box, self.residential_location_box, self.residential_quality_box
        ]


        self._init_ui_layout()  # Setup the layout and specific widget properties

    def _init_ui_layout(self):
        """Initializes the UI layout, widget positions, styles, and connections."""
        self.setObjectName("mainwindow") # For styling

        # --- Folder Selection Area ---
        self.headline_folder_rgb.setGeometry(20, 20, 250, 20)
        self.selected_folder_label_rgb.setGeometry(280, 20, 500, 23)
        self.browse_button_rgb.setGeometry(790, 20, 100, 23) # Increased width
        self.browse_button_rgb.clicked.connect(self.select_rgb_folder)

        self.headline_folder_tir.setGeometry(20, 50, 250, 20)
        self.selected_folder_label_tir.setGeometry(280, 50, 500, 23)
        self.browse_button_tir.setGeometry(790, 50, 100, 23) # Increased width
        self.browse_button_tir.clicked.connect(self.select_tir_folder)

        self.headline_folder_save.setGeometry(20, 80, 250, 20)
        self.selected_folder_label_save.setGeometry(280, 80, 500, 23)
        self.browse_button_save.setGeometry(790, 80, 100, 23) # Increased width
        self.browse_button_save.clicked.connect(self.select_save_folder)

        # --- Image Display Area ---
        self.image_box_rgb.setGeometry(20, 145, self.img_panel_width, self.img_panel_height)
        self.image_box_rgb.setAlignment(Qt.AlignCenter) # Center image
        self.image_box_rgb.setStyleSheet("border: 1px solid #ccc;")
        self.rgb_name_label.setGeometry(20, 120, 300, 20)
        self.progress_bar_rgb.setGeometry(330, 120, 120, 20) # Adjusted position

        self.image_box_tir.setGeometry(20 + self.img_panel_width + 10, 145, self.img_panel_width, self.img_panel_height)
        self.image_box_tir.setAlignment(Qt.AlignCenter)
        self.image_box_tir.setStyleSheet("border: 1px solid #ccc;")
        self.tir_name_label.setGeometry(20 + self.img_panel_width + 10, 120, 300, 20)
        self.progress_bar_tir.setGeometry(330 + self.img_panel_width + 10, 120, 120, 20)

        # --- Annotation Log ---
        self.display_anno_log.setGeometry(20, 590, self.img_panel_width * 2 + 10, 160) # Spans under both images

        # --- Annotation Panel (Scrollable) ---
        X_base_annotation_panel = self.img_panel_width * 2 + 30 # Start x-coordinate for the annotation panel
        annotation_panel_width = 600 # Fixed width for now, adjust as needed
        annotation_panel_height = 700 # Fixed height for now

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(X_base_annotation_panel, 20, annotation_panel_width, annotation_panel_height)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Hide horizontal scrollbar

        self.scroll_content_widget = QWidget() # A QWidget to hold the layout
        self.scroll_area.setWidget(self.scroll_content_widget)
        self.annotation_panel_layout = QVBoxLayout(self.scroll_content_widget) # QVBoxLayout for vertical stacking

        # --- Populate Annotation Panel ---
        self.annotation_panel_layout.addWidget(self.question_headline)

        # Match & Weather
        self.annotation_panel_layout.addWidget(self.chk_match)
        self.match_options_box.addItems(["Select Match", "almost match", "partial match", "not match"])
        self.annotation_panel_layout.addWidget(self.match_options_box)
        self.mist_options_box.addItems(["Select Mist", "mist", "not mist", "not sure"])
        self.annotation_panel_layout.addWidget(self.mist_options_box)
        self.night_options_box.addItems(["Select Darkness", "dark", "not dark", "not sure"])
        self.annotation_panel_layout.addWidget(self.night_options_box)
        self.chk_match.stateChanged.connect(self._update_match_attributes)
        self.match_options_box.currentIndexChanged.connect(self._update_match_attributes)
        self.mist_options_box.currentIndexChanged.connect(self._update_match_attributes)
        self.night_options_box.currentIndexChanged.connect(self._update_match_attributes)


        # Scene Theme
        self.annotation_panel_layout.addWidget(self.chk_theme)
        self.theme_residential_box.addItems(["Residential Area?", "Residential", "n-Residential"])
        self.annotation_panel_layout.addWidget(self.theme_residential_box)
        self.theme_urban_rural_box.addItems(["Urban/Rural?", "Urban", "Rural"])
        self.annotation_panel_layout.addWidget(self.theme_urban_rural_box)
        self.chk_theme.stateChanged.connect(self._update_theme_attributes)
        self.theme_residential_box.currentIndexChanged.connect(self._update_theme_attributes)
        self.theme_urban_rural_box.currentIndexChanged.connect(self._update_theme_attributes)


        # Distance/Location
        self.annotation_panel_layout.addWidget(self.chk_dis_loc)
        self.disloc_pos_a_box.addItems(["Position A", "none", "top", "bottom", "left", "right", "center", "top left", "top right", "bottom left", "bottom right"])
        self.annotation_panel_layout.addWidget(self.disloc_pos_a_box)
        self.disloc_cluster_a_box.addItems(["Is A cluster?", "none", "cluster"])
        self.annotation_panel_layout.addWidget(self.disloc_cluster_a_box)
        # self.subset_list is defined in __init__
        object_categories = ["Object A"] + self.LC2SubList["building"][1:] + self.LC2SubList["vegetation area"][1:] + self.LC2SubList["water area"][1:] + self.LC2SubList["road"][1:] + ["agricultural area", "wasteland", "intersection", "parking area", "park", "concrete floor"] + self.LC2SubList["sports field"][1:] + ["pier", "beach", "airport", "apron"]
        self.disloc_obj_a_box.addItems(list(dict.fromkeys(object_categories))) # Remove duplicates
        self.annotation_panel_layout.addWidget(self.disloc_obj_a_box)

        self.disloc_pos_b_box.addItems(["Position B", "none", "top", "bottom", "left", "right", "center", "top left", "top right", "bottom left", "bottom right"])
        self.annotation_panel_layout.addWidget(self.disloc_pos_b_box)
        self.disloc_cluster_b_box.addItems(["Is B cluster?", "none", "cluster"])
        self.annotation_panel_layout.addWidget(self.disloc_cluster_b_box)
        object_categories_b = ["Object B"] + self.LC2SubList["building"][1:] + self.LC2SubList["vegetation area"][1:] + self.LC2SubList["water area"][1:] + self.LC2SubList["road"][1:] + ["agricultural area", "wasteland", "intersection", "parking area", "park", "concrete floor"] + self.LC2SubList["sports field"][1:] + ["pier", "beach", "airport", "apron"]
        self.disloc_obj_b_box.addItems(list(dict.fromkeys(object_categories_b)))
        self.annotation_panel_layout.addWidget(self.disloc_obj_b_box)

        self.disloc_distance_box.addItems(["A-B Distance (units)", "none", "next to", "0-25", "25-50", "50-75", "75-100", "100-125", "125-150", "150-175", "175-200", "200+"])
        self.annotation_panel_layout.addWidget(self.disloc_distance_box)
        self.disloc_relation_box.addItems(["B's pos relative to A"] + self.location_options)
        self.annotation_panel_layout.addWidget(self.disloc_relation_box)
        self.annotation_panel_layout.addWidget(self.btn_measure_distance)
        self.annotation_panel_layout.addWidget(self.btn_submit_disloc)
        self.annotation_panel_layout.addWidget(self.btn_finish_disloc)
        self.chk_dis_loc.stateChanged.connect(self._handle_disloc_state_change)
        self.btn_submit_disloc.clicked.connect(self._confirm_disloc_pair)
        self.btn_measure_distance.clicked.connect(self.trigger_distance_measurement)
        self.btn_finish_disloc.clicked.connect(self._finalize_disloc_attributes)


        # Contain/Presence
        self.annotation_panel_layout.addWidget(self.chk_contain)
        initial_landcovers = ["LandCover Class"] + list(self.LC2SubList.keys())[:-1] # Exclude the placeholder
        self.contain_landcover_box.addItems(initial_landcovers)
        self.annotation_panel_layout.addWidget(self.contain_landcover_box)
        self.contain_subset_box.addItems(["Subset Class"]) # Populated dynamically
        self.annotation_panel_layout.addWidget(self.contain_subset_box)
        self.contain_number_box.addItems(["Number", "none", "1", "2", "3", "4", "5", "6", "6-10", "10-20", "20-40", "40-100", "> 100"])
        self.annotation_panel_layout.addWidget(self.contain_number_box)
        self.contain_location_box.addItems(["Location", "none", "top", "bottom", "left", "right", "center", "upper left", "upper right", "lower left", "lower right", "almost all the picture"])
        self.annotation_panel_layout.addWidget(self.contain_location_box)
        self.contain_shape_box.addItems(["Shape", "none", "Straight", "Curved", "Triangle", "Square", "Rectangle", "other quadrilater", "Rotundity", "other shape"])
        self.annotation_panel_layout.addWidget(self.contain_shape_box)
        self.contain_area_box.addItems(["Area (units^2)", "none", "0-100", "100-500", "500-1000", "1000-5000", "5000-10000", "10000+"]) # Simplified
        self.annotation_panel_layout.addWidget(self.contain_area_box)
        self.contain_length_box.addItems(["Length (units)", "none", "0-25", "25-50", "50-100", "100-200", "200+"]) # Simplified
        self.annotation_panel_layout.addWidget(self.contain_length_box)
        self.contain_distribution_box.addItems(["Distribution", "none", "Clustered", "Isolated", "Dense", "Random", "Uniform"])
        self.annotation_panel_layout.addWidget(self.contain_distribution_box)
        self.contain_quality_box.addItems(["Image Quality", "optical", "thermal", "almost same"])
        self.annotation_panel_layout.addWidget(self.contain_quality_box)
        self.annotation_panel_layout.addWidget(self.btn_measure_area)
        self.annotation_panel_layout.addWidget(self.btn_submit_contain)
        self.annotation_panel_layout.addWidget(self.btn_delete_contain_landcover)
        self.annotation_panel_layout.addWidget(self.btn_finish_contain)
        self.chk_contain.stateChanged.connect(self._handle_contain_state_change)
        self.contain_landcover_box.currentIndexChanged.connect(self._populate_contain_subset_box)
        self.contain_landcover_box.currentIndexChanged.connect(self._update_current_landcover_for_delete_contain)
        self.contain_subset_box.currentIndexChanged.connect(self._load_existing_contain_attributes)
        self.btn_submit_contain.clicked.connect(self._confirm_contain_item)
        self.btn_measure_area.clicked.connect(self.trigger_area_measurement)
        self.btn_delete_contain_landcover.clicked.connect(self._delete_single_contain_landcover_attributes)
        self.btn_finish_contain.clicked.connect(self._finalize_contain_attributes)


        # Deduce (Custom Attributes)
        self.annotation_panel_layout.addWidget(self.chk_deduce)
        self.annotation_panel_layout.addWidget(self.deduce_q1_input)
        self.annotation_panel_layout.addWidget(self.deduce_a1_input)
        self.annotation_panel_layout.addWidget(self.btn_submit_deduce1)
        self.annotation_panel_layout.addWidget(self.deduce_q2_input)
        self.annotation_panel_layout.addWidget(self.deduce_a2_input)
        self.annotation_panel_layout.addWidget(self.btn_submit_deduce2)
        self.chk_deduce.stateChanged.connect(self._handle_deduce_state_change)
        self.btn_submit_deduce1.clicked.connect(self._update_deduce_attribute_one)
        self.btn_submit_deduce2.clicked.connect(self._update_deduce_attribute_two)


        # Traffic Elements
        self.annotation_panel_layout.addWidget(self.chk_traffic)
        initial_traffic_landcovers = ["Traffic Category"] + list(self.TR2ObjList.keys())[:-1]
        self.traffic_landcover_box.addItems(initial_traffic_landcovers)
        self.annotation_panel_layout.addWidget(self.traffic_landcover_box)
        self.traffic_object_box.addItems(["Traffic Object"]) # Populated dynamically
        self.annotation_panel_layout.addWidget(self.traffic_object_box)
        self.traffic_number_box.addItems(["Number", "none", "1", "2", "3", "4", "5", "6", "6-10", "10-20", "20-40", ">40"])
        self.annotation_panel_layout.addWidget(self.traffic_number_box)
        self.traffic_location_box.addItems(["Location", "none", "top", "bottom", "left", "right", "center", "multiple"])
        self.annotation_panel_layout.addWidget(self.traffic_location_box)
        self.traffic_quality_box.addItems(["Image Quality", "optical", "thermal", "almost same"])
        self.annotation_panel_layout.addWidget(self.traffic_quality_box)
        self.annotation_panel_layout.addWidget(self.btn_submit_traffic)
        self.annotation_panel_layout.addWidget(self.btn_delete_traffic_landcover)
        self.annotation_panel_layout.addWidget(self.btn_finish_traffic)
        self.chk_traffic.stateChanged.connect(self._handle_traffic_state_change)
        self.traffic_landcover_box.currentIndexChanged.connect(self._populate_traffic_object_box)
        self.traffic_landcover_box.currentIndexChanged.connect(self._update_current_landcover_for_delete_traffic)
        self.traffic_object_box.currentIndexChanged.connect(self._load_existing_traffic_attributes)
        self.btn_submit_traffic.clicked.connect(self._confirm_traffic_item)
        self.btn_delete_traffic_landcover.clicked.connect(self._delete_single_traffic_landcover_attributes)
        self.btn_finish_traffic.clicked.connect(self._finalize_traffic_attributes)


        # Residential Elements
        self.annotation_panel_layout.addWidget(self.chk_residential)
        initial_residential_landcovers = ["Residential Category"] + list(self.RA2ObjList.keys())[:-1]
        self.residential_landcover_box.addItems(initial_residential_landcovers)
        self.annotation_panel_layout.addWidget(self.residential_landcover_box)
        self.residential_object_box.addItems(["Residential Object"]) # Populated dynamically
        self.annotation_panel_layout.addWidget(self.residential_object_box)
        self.residential_number_box.addItems(["Number", "none", "1", "2", "3", "4", "5", "6-10", ">10"])
        self.annotation_panel_layout.addWidget(self.residential_number_box)
        self.residential_location_box.addItems(["Location", "none", "top", "bottom", "left", "right", "center", "multiple"])
        self.annotation_panel_layout.addWidget(self.residential_location_box)
        self.residential_quality_box.addItems(["Image Quality", "optical", "thermal", "almost same"])
        self.annotation_panel_layout.addWidget(self.residential_quality_box)
        self.annotation_panel_layout.addWidget(self.btn_submit_residential)
        self.annotation_panel_layout.addWidget(self.btn_delete_residential_landcover)
        self.annotation_panel_layout.addWidget(self.btn_finish_residential)
        self.chk_residential.stateChanged.connect(self._handle_residential_state_change)
        self.residential_landcover_box.currentIndexChanged.connect(self._populate_residential_object_box)
        self.residential_landcover_box.currentIndexChanged.connect(self._update_current_landcover_for_delete_residential)
        self.residential_object_box.currentIndexChanged.connect(self._load_existing_residential_attributes)
        self.btn_submit_residential.clicked.connect(self._confirm_residential_item)
        self.btn_delete_residential_landcover.clicked.connect(self._delete_single_residential_landcover_attributes)
        self.btn_finish_residential.clicked.connect(self._finalize_residential_attributes)


        # Agricultural Features
        self.annotation_panel_layout.addWidget(self.chk_agricultural)
        self.agri_road_box.addItems(["Agricultural Road?", "Yes", "No", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.agri_road_box)
        self.agri_water_box.addItems(["Agricultural Water?", "Yes", "No", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.agri_water_box)
        self.chk_agricultural.stateChanged.connect(self._update_agricultural_attributes)
        self.agri_road_box.currentIndexChanged.connect(self._update_agricultural_attributes)
        self.agri_water_box.currentIndexChanged.connect(self._update_agricultural_attributes)


        # Industrial Features
        self.annotation_panel_layout.addWidget(self.chk_industrial)
        self.ind_facility_box.addItems(["Industrial Facility?", "Yes", "No", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.ind_facility_box)
        self.ind_scale_box.addItems(["Industrial Scale", "small", "medium", "large", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.ind_scale_box)
        self.ind_location_box.addItems(["Industrial Location"] + self.location_options + ["center", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.ind_location_box)
        self.chk_industrial.stateChanged.connect(self._update_industrial_attributes)
        self.ind_facility_box.currentIndexChanged.connect(self._update_industrial_attributes)
        self.ind_scale_box.currentIndexChanged.connect(self._update_industrial_attributes)
        self.ind_location_box.currentIndexChanged.connect(self._update_industrial_attributes)


        # UAV Parameters
        self.annotation_panel_layout.addWidget(self.chk_uav)
        self.uav_height_box.addItems(["UAV Height (m)", "150-250", "250-400", "400-550", "none", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.uav_height_box)
        self.uav_angle_box.addItems(["UAV Angle", "vertical", "oblique", "Not Applicable"])
        self.annotation_panel_layout.addWidget(self.uav_angle_box)
        self.chk_uav.stateChanged.connect(self._update_uav_attributes)
        self.uav_height_box.currentIndexChanged.connect(self._update_uav_attributes)
        self.uav_angle_box.currentIndexChanged.connect(self._update_uav_attributes)


        # --- Navigation and File Operations Buttons (bottom of scroll panel or fixed area) ---
        # For simplicity, adding them to the scroll layout. For fixed position, place outside scroll_area.
        self.annotation_panel_layout.addSpacing(30) # Add some space before final buttons
        self.annotation_panel_layout.addWidget(self.btn_previous_image)
        self.annotation_panel_layout.addWidget(self.btn_next_image)
        self.annotation_panel_layout.addWidget(self.btn_submit_all_current)
        self.annotation_panel_layout.addWidget(self.btn_generate_file)

        self.btn_previous_image.clicked.connect(self.show_previous_image)
        self.btn_next_image.clicked.connect(self.show_next_image)
        self.btn_submit_all_current.clicked.connect(self.save_current_image_annotations_to_memory)
        self.btn_generate_file.clicked.connect(self.save_all_annotations_to_file)


        # --- Styling ---
        # Apply some basic styling for better appearance
        common_button_style = "font-size: 14px; padding: 5px;"
        self.browse_button_rgb.setStyleSheet(common_button_style)
        self.browse_button_tir.setStyleSheet(common_button_style)
        self.browse_button_save.setStyleSheet(common_button_style)
        self.btn_previous_image.setStyleSheet(common_button_style + "font-weight:bold;")
        self.btn_next_image.setStyleSheet(common_button_style + "font-weight:bold;")
        self.btn_submit_all_current.setStyleSheet(common_button_style + "background-color: #D3E3FD;") # Light blue
        self.btn_generate_file.setStyleSheet(common_button_style + "background-color: #C8E6C9; font-weight:bold;") # Light green

        label_style = "font-weight: bold; font-size: 13px;"
        self.headline_folder_rgb.setStyleSheet(label_style)
        self.headline_folder_tir.setStyleSheet(label_style)
        self.headline_folder_save.setStyleSheet(label_style)
        self.question_headline.setStyleSheet(label_style + "font-size: 15px; margin-bottom: 10px;")

        self.selected_folder_label_rgb.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 2px;")
        self.selected_folder_label_tir.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 2px;")
        self.selected_folder_label_save.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 2px;")

        checkbox_style = "font-size: 14px; font-weight: bold; margin-top: 10px; margin-bottom: 5px;"
        for chk_box in self.all_checkboxes:
            chk_box.setStyleSheet(checkbox_style)

        combobox_style = "font-size: 13px; background-color: white; padding: 3px;"
        lineedit_style = "font-size: 13px; padding: 3px;"
        for widget in self.all_comboboxes_and_lineedits:
            if isinstance(widget, QComboBox):
                widget.setStyleSheet(combobox_style)
            elif isinstance(widget, QLineEdit):
                widget.setStyleSheet(lineedit_style)
        # Specific button styles for annotation panel
        action_button_style = "font-size: 13px; padding: 4px; margin-top: 5px;"
        self.btn_submit_disloc.setStyleSheet(action_button_style)
        self.btn_measure_distance.setStyleSheet(action_button_style)
        self.btn_finish_disloc.setStyleSheet(action_button_style + "background-color: #FFECB3;") # Light yellow
        self.btn_submit_contain.setStyleSheet(action_button_style)
        self.btn_measure_area.setStyleSheet(action_button_style)
        self.btn_finish_contain.setStyleSheet(action_button_style + "background-color: #FFECB3;")
        self.btn_delete_contain_landcover.setStyleSheet(action_button_style + "background-color: #FFCDD2;") # Light red
        self.btn_submit_deduce1.setStyleSheet(action_button_style)
        self.btn_submit_deduce2.setStyleSheet(action_button_style)
        self.btn_submit_traffic.setStyleSheet(action_button_style)
        self.btn_finish_traffic.setStyleSheet(action_button_style + "background-color: #FFECB3;")
        self.btn_delete_traffic_landcover.setStyleSheet(action_button_style + "background-color: #FFCDD2;")
        self.btn_submit_residential.setStyleSheet(action_button_style)
        self.btn_finish_residential.setStyleSheet(action_button_style + "background-color: #FFECB3;")
        self.btn_delete_residential_landcover.setStyleSheet(action_button_style + "background-color: #FFCDD2;")


    # --- Core Annotation Logic ---
    def _add_or_update_attribute(self, key, value):
        """Adds or updates an attribute in the current_image_attributes dictionary."""
        if self.allow_modification:
            if value is not None and value != "": # Allow empty string if intended
                self.current_image_attributes[key] = value
            elif key in self.current_image_attributes: # If value is None or empty, remove if exists
                del self.current_image_attributes[key]
            self._display_current_attributes_in_log()

    def _remove_attribute(self, key):
        """Removes an attribute if it exists."""
        if self.allow_modification and key in self.current_image_attributes:
            del self.current_image_attributes[key]
            self._display_current_attributes_in_log()

    def _display_current_attributes_in_log(self):
        """Displays the currently collected attributes in the annotation log."""
        self.display_anno_log.clear()
        if not self.current_image_attributes:
            self.display_anno_log.setText("No attributes selected for the current image.")
            return

        log_text = "Current Image Attributes:\n"
        for key, value in sorted(self.current_image_attributes.items()):
            if isinstance(value, dict): # For complex attributes like PresContain, Traffic
                log_text += f"  {key}:\n"
                for sub_key, sub_value_list in value.items():
                    log_text += f"    - {sub_key}:\n"
                    for item_attrs in sub_value_list: # item_attrs is a list of [index, text]
                        # Format this more nicely, e.g., "AttributeName: Value"
                        attr_strings = [f"{self._get_attr_name_for_log(key, idx, item_attrs[0])}: {item_attrs[1]}" for idx, item_attrs in enumerate(item_attrs)]
                        log_text += f"        {', '.join(attr_strings)}\n"
            elif isinstance(value, list) and key == "LocDis": # Special handling for LocDis list of lists
                 log_text += f"  {key}:\n"
                 for pair_attrs in value:
                    # pair_attrs is [objA_text, objB_text, distance_idx, location_text, distance_text]
                    log_text += f"    - Pair: ({pair_attrs[0]}) & ({pair_attrs[1]}), Loc: {pair_attrs[3]}, Dist: {pair_attrs[4]}\n"

            else:
                log_text += f"  {key}: {value}\n"
        self.display_anno_log.setText(log_text)

    def _get_attr_name_for_log(self, main_category, attr_index, attr_value_index):
        """ Helper to get descriptive names for attributes in the log for complex types. """
        # This needs to be customized based on the order of your QComboBoxes for each category
        if main_category == "PresContain":
            names = ["Subset", "Number", "Location", "Shape", "Area", "Length", "Distribution", "Quality"]
            return names[attr_index] if attr_index < len(names) else f"Attr_{attr_index+1}"
        elif main_category == "Traffic":
            names = ["Object", "Number", "Location", "Quality"]
            return names[attr_index] if attr_index < len(names) else f"Attr_{attr_index+1}"
        elif main_category == "Residential":
            names = ["Object", "Number", "Location", "Quality"]
            return names[attr_index] if attr_index < len(names) else f"Attr_{attr_index+1}"
        return f"Attr_{attr_index+1}"


    # --- Category-Specific Attribute Handling Callbacks ---
    def _update_match_attributes(self):
        if self.chk_match.isChecked():
            if self.match_options_box.currentIndex() > 0:
                self._add_or_update_attribute("match_condition", self.match_options_box.currentText())
            else:
                self._remove_attribute("match_condition")

            if self.mist_options_box.currentIndex() > 0:
                self._add_or_update_attribute("mist_condition", self.mist_options_box.currentText())
            else:
                self._remove_attribute("mist_condition")

            if self.night_options_box.currentIndex() > 0:
                self._add_or_update_attribute("darkness_condition", self.night_options_box.currentText())
            else:
                self._remove_attribute("darkness_condition")
        else:
            self._remove_attribute("match_condition")
            self._remove_attribute("mist_condition")
            self._remove_attribute("darkness_condition")

    def _update_theme_attributes(self):
        if self.chk_theme.isChecked():
            if self.theme_residential_box.currentIndex() > 0:
                self._add_or_update_attribute("area_type", self.theme_residential_box.currentText())
            else:
                self._remove_attribute("area_type")
            if self.theme_urban_rural_box.currentIndex() > 0:
                self._add_or_update_attribute("scene_macro_category", self.theme_urban_rural_box.currentText())
            else:
                self._remove_attribute("scene_macro_category")
        else:
            self._remove_attribute("area_type")
            self._remove_attribute("scene_macro_category")

    def _handle_disloc_state_change(self):
        if not self.chk_dis_loc.isChecked():
            self._remove_attribute("LocDis")
            self.dis_loc_details = {} # Clear temporary storage
            self._display_current_attributes_in_log() # Update log

    def _all_disloc_options_selected(self):
        """Checks if all required Distance/Location options for a pair are selected."""
        return (self.disloc_pos_a_box.currentIndex() > 0 and
                self.disloc_obj_a_box.currentIndex() > 0 and
                self.disloc_cluster_a_box.currentIndex() > 0 and # Added check for cluster
                self.disloc_pos_b_box.currentIndex() > 0 and
                self.disloc_obj_b_box.currentIndex() > 0 and
                self.disloc_cluster_b_box.currentIndex() > 0 and # Added check for cluster
                self.disloc_distance_box.currentIndex() > 0 and
                self.disloc_relation_box.currentIndex() > 0)

    def _confirm_disloc_pair(self):
        if self.chk_dis_loc.isChecked() and self._all_disloc_options_selected():
            obj_a_base = self.disloc_obj_a_box.currentText()
            obj_a_pos = self.disloc_pos_a_box.currentText()
            obj_a_cluster = self.disloc_cluster_a_box.currentText()
            obj_a_text = f"{'a cluster of ' if obj_a_cluster == 'cluster' else ''}{obj_a_base}"
            if obj_a_pos != "none":
                obj_a_text += f" at the {obj_a_pos} of the picture"

            obj_b_base = self.disloc_obj_b_box.currentText()
            obj_b_pos = self.disloc_pos_b_box.currentText()
            obj_b_cluster = self.disloc_cluster_b_box.currentText()
            obj_b_text = f"{'a cluster of ' if obj_b_cluster == 'cluster' else ''}{obj_b_base}"
            if obj_b_pos != "none":
                obj_b_text += f" at the {obj_b_pos} of the picture"

            pair_key = (obj_a_text, obj_b_text)
            self.dis_loc_details[pair_key] = [
                self.disloc_distance_box.currentIndex(),
                self.disloc_relation_box.currentText(),
                self.disloc_distance_box.currentText()
            ]
            self._show_temporary_complex_annotation_log("LocDis")
            # Optionally reset D/L comboboxes for next pair entry
            # self.disloc_pos_a_box.setCurrentIndex(0) ... etc.
        else:
            self._show_warning("Please select all options for the Distance/Location pair.")

    def _finalize_disloc_attributes(self):
        if self.chk_dis_loc.isChecked() and self.dis_loc_details:
            # Convert dict to list of lists for consistent storage
            locdis_list_for_storage = []
            for (obj_a, obj_b), details in self.dis_loc_details.items():
                locdis_list_for_storage.append([obj_a, obj_b] + details)
            self._add_or_update_attribute("LocDis", locdis_list_for_storage)
            # self.dis_loc_details = {} # Clear after finalizing if needed, or keep for modification
        elif self.chk_dis_loc.isChecked() and not self.dis_loc_details:
            self._remove_attribute("LocDis") # If checkbox is on but no details, remove old entry
        self._display_current_attributes_in_log()


    def _handle_contain_state_change(self):
        if not self.chk_contain.isChecked():
            self._remove_attribute("PresContain")
            self.contain_details = {}
            self._display_current_attributes_in_log()

    def _populate_contain_subset_box(self):
        landcover = self.contain_landcover_box.currentText()
        self.contain_subset_box.clear()
        if landcover in self.LC2SubList:
            self.contain_subset_box.addItems(self.LC2SubList[landcover])
        else:
            self.contain_subset_box.addItems(["Subset Class"]) # Default if no specific subsets

    def _update_current_landcover_for_delete_contain(self):
        if self.contain_landcover_box.currentIndex() > 0:
            self.current_landcover_for_delete = self.contain_landcover_box.currentText()
        else:
            self.current_landcover_for_delete = "None"

    def _all_contain_options_selected(self):
        return (self.contain_landcover_box.currentIndex() > 0 and
                self.contain_subset_box.currentIndex() > 0 and
                self.contain_number_box.currentIndex() > 0 and
                self.contain_location_box.currentIndex() > 0 and
                self.contain_shape_box.currentIndex() > 0 and
                self.contain_area_box.currentIndex() > 0 and
                self.contain_length_box.currentIndex() > 0 and
                self.contain_distribution_box.currentIndex() > 0 and
                self.contain_quality_box.currentIndex() > 0)

    def _confirm_contain_item(self):
        if self.chk_contain.isChecked() and self._all_contain_options_selected():
            landcover = self.contain_landcover_box.currentText()
            item_attributes = []
            for widget in self.contain_attribute_widgets: # Use the defined list
                item_attributes.append([widget.currentIndex(), widget.currentText()])

            if landcover not in self.contain_details:
                self.contain_details[landcover] = []

            # Check if this subset already exists for this landcover, if so, update it
            subset_to_add_or_update = self.contain_subset_box.currentText()
            found_and_updated = False
            for i, existing_item_attrs in enumerate(self.contain_details[landcover]):
                if existing_item_attrs[0][1] == subset_to_add_or_update: # existing_item_attrs[0] is [subset_idx, subset_text]
                    self.contain_details[landcover][i] = item_attributes
                    found_and_updated = True
                    break
            if not found_and_updated:
                self.contain_details[landcover].append(item_attributes)

            self._show_temporary_complex_annotation_log("PresContain")
            # Optionally reset C/P comboboxes for next item (except landcover)
            # for widget in self.contain_attribute_widgets: widget.setCurrentIndex(0)
        else:
            self. _show_warning("Please select all options for the Contain/Presence item.")

    def _load_existing_contain_attributes(self):
        if self.allow_modification and self.chk_contain.isChecked() and \
           self.contain_landcover_box.currentIndex() > 0 and \
           self.contain_subset_box.currentIndex() > 0:

            landcover = self.contain_landcover_box.currentText()
            subset_text = self.contain_subset_box.currentText()

            if landcover in self.contain_details:
                for item_attrs_list in self.contain_details[landcover]:
                    # item_attrs_list is like [[idx_subset, text_subset], [idx_num, text_num], ...]
                    if item_attrs_list[0][1] == subset_text: # Check if subset text matches
                        self.allow_modification = False # Disable updates while loading
                        for i, widget in enumerate(self.contain_attribute_widgets):
                            # item_attrs_list[i] is [index, text] for the current attribute
                            widget.setCurrentIndex(item_attrs_list[i][0])
                        self.allow_modification = True # Re-enable
                        break # Found and loaded

    def _delete_single_contain_landcover_attributes(self):
        if self.chk_contain.isChecked() and self.current_landcover_for_delete != "None":
            if self.current_landcover_for_delete in self.contain_details:
                del self.contain_details[self.current_landcover_for_delete]
                self._show_temporary_complex_annotation_log("PresContain") # Update log
                # Also update the main attribute dict if it was already finalized
                if "PresContain" in self.current_image_attributes and \
                   self.current_landcover_for_delete in self.current_image_attributes["PresContain"]:
                    del self.current_image_attributes["PresContain"][self.current_landcover_for_delete]
                    if not self.current_image_attributes["PresContain"]: # If dict becomes empty
                        del self.current_image_attributes["PresContain"]
                    self._display_current_attributes_in_log()

    def _finalize_contain_attributes(self):
        if self.chk_contain.isChecked() and self.contain_details:
            self._add_or_update_attribute("PresContain", dict(self.contain_details)) # Store a copy
        elif self.chk_contain.isChecked() and not self.contain_details:
             self._remove_attribute("PresContain")
        self._display_current_attributes_in_log()


    def _handle_deduce_state_change(self):
        if not self.chk_deduce.isChecked():
            # Remove potentially added custom attributes
            if self.deduce_q1_input.property("custom_key"): # Check if a key was set
                 self._remove_attribute(self.deduce_q1_input.property("custom_key"))
            if self.deduce_q2_input.property("custom_key"):
                 self._remove_attribute(self.deduce_q2_input.property("custom_key"))
            self.deduce_q1_input.setText("Enter custom attribute 1 key")
            self.deduce_a1_input.setText("Enter attribute 1 value")
            self.deduce_q2_input.setText("Enter custom attribute 2 key")
            self.deduce_a2_input.setText("Enter attribute 2 value")
            self.deduce_q1_input.setProperty("custom_key", None) # Clear stored key
            self.deduce_q2_input.setProperty("custom_key", None)

    def _update_deduce_attribute_one(self):
        if self.chk_deduce.isChecked():
            key = self.deduce_q1_input.text().strip()
            value = self.deduce_a1_input.text().strip()
            if key and key != "Enter custom attribute 1 key":
                self._add_or_update_attribute(key, value if value != "Enter attribute 1 value" else "")
                self.deduce_q1_input.setProperty("custom_key", key) # Store the key for potential removal
            else:
                self._show_warning("Please enter a valid key for custom attribute 1.")

    def _update_deduce_attribute_two(self):
        if self.chk_deduce.isChecked():
            key = self.deduce_q2_input.text().strip()
            value = self.deduce_a2_input.text().strip()
            if key and key != "Enter custom attribute 2 key":
                self._add_or_update_attribute(key, value if value != "Enter attribute 2 value" else "")
                self.deduce_q2_input.setProperty("custom_key", key)
            else:
                self._show_warning("Please enter a valid key for custom attribute 2.")


    def _handle_traffic_state_change(self):
        if not self.chk_traffic.isChecked():
            self._remove_attribute("Traffic")
            self.traffic_details = {}
            self._display_current_attributes_in_log()

    def _populate_traffic_object_box(self):
        landcover = self.traffic_landcover_box.currentText()
        self.traffic_object_box.clear()
        if landcover in self.TR2ObjList:
            self.traffic_object_box.addItems(self.TR2ObjList[landcover])
        else:
            self.traffic_object_box.addItems(["Traffic Object"])

    def _update_current_landcover_for_delete_traffic(self):
        if self.traffic_landcover_box.currentIndex() > 0:
            self.current_traffic_landcover_for_delete = self.traffic_landcover_box.currentText()
        else:
            self.current_traffic_landcover_for_delete = "None"

    def _all_traffic_options_selected(self):
         return (self.traffic_landcover_box.currentIndex() > 0 and
                self.traffic_object_box.currentIndex() > 0 and
                self.traffic_number_box.currentIndex() > 0 and
                self.traffic_location_box.currentIndex() > 0 and
                self.traffic_quality_box.currentIndex() > 0)

    def _confirm_traffic_item(self):
        if self.chk_traffic.isChecked() and self._all_traffic_options_selected():
            landcover = self.traffic_landcover_box.currentText()
            item_attributes = []
            for widget in self.traffic_attribute_widgets:
                item_attributes.append([widget.currentIndex(), widget.currentText()])

            if landcover not in self.traffic_details:
                self.traffic_details[landcover] = []

            object_to_add_or_update = self.traffic_object_box.currentText()
            found_and_updated = False
            for i, existing_item_attrs in enumerate(self.traffic_details[landcover]):
                if existing_item_attrs[0][1] == object_to_add_or_update:
                    self.traffic_details[landcover][i] = item_attributes
                    found_and_updated = True
                    break
            if not found_and_updated:
                self.traffic_details[landcover].append(item_attributes)
            self._show_temporary_complex_annotation_log("Traffic")
        else:
            self._show_warning("Please select all options for the Traffic item.")

    def _load_existing_traffic_attributes(self):
        if self.allow_modification and self.chk_traffic.isChecked() and \
           self.traffic_landcover_box.currentIndex() > 0 and \
           self.traffic_object_box.currentIndex() > 0:
            landcover = self.traffic_landcover_box.currentText()
            object_text = self.traffic_object_box.currentText()
            if landcover in self.traffic_details:
                for item_attrs_list in self.traffic_details[landcover]:
                    if item_attrs_list[0][1] == object_text:
                        self.allow_modification = False
                        for i, widget in enumerate(self.traffic_attribute_widgets):
                            widget.setCurrentIndex(item_attrs_list[i][0])
                        self.allow_modification = True
                        break

    def _delete_single_traffic_landcover_attributes(self):
        if self.chk_traffic.isChecked() and self.current_traffic_landcover_for_delete != "None":
            if self.current_traffic_landcover_for_delete in self.traffic_details:
                del self.traffic_details[self.current_traffic_landcover_for_delete]
                self._show_temporary_complex_annotation_log("Traffic")
                if "Traffic" in self.current_image_attributes and \
                   self.current_traffic_landcover_for_delete in self.current_image_attributes["Traffic"]:
                    del self.current_image_attributes["Traffic"][self.current_traffic_landcover_for_delete]
                    if not self.current_image_attributes["Traffic"]:
                        del self.current_image_attributes["Traffic"]
                    self._display_current_attributes_in_log()

    def _finalize_traffic_attributes(self):
        if self.chk_traffic.isChecked() and self.traffic_details:
            self._add_or_update_attribute("Traffic", dict(self.traffic_details))
        elif self.chk_traffic.isChecked() and not self.traffic_details:
            self._remove_attribute("Traffic")
        self._display_current_attributes_in_log()


    def _handle_residential_state_change(self):
        if not self.chk_residential.isChecked():
            self._remove_attribute("Residential")
            self.residential_details = {}
            self._display_current_attributes_in_log()

    def _populate_residential_object_box(self):
        landcover = self.residential_landcover_box.currentText()
        self.residential_object_box.clear()
        if landcover in self.RA2ObjList:
            self.residential_object_box.addItems(self.RA2ObjList[landcover])
        else:
            self.residential_object_box.addItems(["Residential Object"])

    def _update_current_landcover_for_delete_residential(self):
        if self.residential_landcover_box.currentIndex() > 0:
            self.current_residential_landcover_for_delete = self.residential_landcover_box.currentText()
        else:
            self.current_residential_landcover_for_delete = "None"

    def _all_residential_options_selected(self):
        return (self.residential_landcover_box.currentIndex() > 0 and
                self.residential_object_box.currentIndex() > 0 and
                self.residential_number_box.currentIndex() > 0 and
                self.residential_location_box.currentIndex() > 0 and
                self.residential_quality_box.currentIndex() > 0)

    def _confirm_residential_item(self):
        if self.chk_residential.isChecked() and self._all_residential_options_selected():
            landcover = self.residential_landcover_box.currentText()
            item_attributes = []
            for widget in self.residential_attribute_widgets:
                item_attributes.append([widget.currentIndex(), widget.currentText()])

            if landcover not in self.residential_details:
                self.residential_details[landcover] = []

            object_to_add_or_update = self.residential_object_box.currentText()
            found_and_updated = False
            for i, existing_item_attrs in enumerate(self.residential_details[landcover]):
                if existing_item_attrs[0][1] == object_to_add_or_update:
                    self.residential_details[landcover][i] = item_attributes
                    found_and_updated = True
                    break
            if not found_and_updated:
                self.residential_details[landcover].append(item_attributes)
            self._show_temporary_complex_annotation_log("Residential")
        else:
            self._show_warning("Please select all options for the Residential item.")


    def _load_existing_residential_attributes(self):
        if self.allow_modification and self.chk_residential.isChecked() and \
           self.residential_landcover_box.currentIndex() > 0 and \
           self.residential_object_box.currentIndex() > 0:
            landcover = self.residential_landcover_box.currentText()
            object_text = self.residential_object_box.currentText()
            if landcover in self.residential_details:
                for item_attrs_list in self.residential_details[landcover]:
                    if item_attrs_list[0][1] == object_text:
                        self.allow_modification = False
                        for i, widget in enumerate(self.residential_attribute_widgets):
                            widget.setCurrentIndex(item_attrs_list[i][0])
                        self.allow_modification = True
                        break

    def _delete_single_residential_landcover_attributes(self):
        if self.chk_residential.isChecked() and self.current_residential_landcover_for_delete != "None":
            if self.current_residential_landcover_for_delete in self.residential_details:
                del self.residential_details[self.current_residential_landcover_for_delete]
                self._show_temporary_complex_annotation_log("Residential")
                if "Residential" in self.current_image_attributes and \
                   self.current_residential_landcover_for_delete in self.current_image_attributes["Residential"]:
                    del self.current_image_attributes["Residential"][self.current_residential_landcover_for_delete]
                    if not self.current_image_attributes["Residential"]:
                        del self.current_image_attributes["Residential"]
                    self._display_current_attributes_in_log()

    def _finalize_residential_attributes(self):
        if self.chk_residential.isChecked() and self.residential_details:
            self._add_or_update_attribute("Residential", dict(self.residential_details))
        elif self.chk_residential.isChecked() and not self.residential_details:
            self._remove_attribute("Residential")
        self._display_current_attributes_in_log()


    def _update_agricultural_attributes(self):
        if self.chk_agricultural.isChecked():
            if self.agri_road_box.currentIndex() > 0: # 0 is placeholder "Agricultural Road?"
                self._add_or_update_attribute("agricultural_road", self.agri_road_box.currentText())
            else:
                self._remove_attribute("agricultural_road")
            if self.agri_water_box.currentIndex() > 0: # 0 is placeholder "Agricultural Water?"
                self._add_or_update_attribute("agricultural_water", self.agri_water_box.currentText())
            else:
                self._remove_attribute("agricultural_water")
        else:
            self._remove_attribute("agricultural_road")
            self._remove_attribute("agricultural_water")

    def _update_industrial_attributes(self):
        if self.chk_industrial.isChecked():
            if self.ind_facility_box.currentIndex() > 0:
                self._add_or_update_attribute("industrial_facility", self.ind_facility_box.currentText())
            else:
                self._remove_attribute("industrial_facility")
            if self.ind_scale_box.currentIndex() > 0:
                self._add_or_update_attribute("industrial_scale", self.ind_scale_box.currentText())
            else:
                self._remove_attribute("industrial_scale")
            if self.ind_location_box.currentIndex() > 0:
                self._add_or_update_attribute("industrial_location", self.ind_location_box.currentText())
            else:
                self._remove_attribute("industrial_location")
        else:
            self._remove_attribute("industrial_facility")
            self._remove_attribute("industrial_scale")
            self._remove_attribute("industrial_location")

    def _update_uav_attributes(self):
        if self.chk_uav.isChecked():
            if self.uav_height_box.currentIndex() > 0:
                self._add_or_update_attribute("uav_height", self.uav_height_box.currentText())
            else:
                self._remove_attribute("uav_height")
            if self.uav_angle_box.currentIndex() > 0:
                self._add_or_update_attribute("uav_angle", self.uav_angle_box.currentText())
            else:
                self._remove_attribute("uav_angle")
        else:
            self._remove_attribute("uav_height")
            self._remove_attribute("uav_angle")


    def _show_temporary_complex_annotation_log(self, category_key):
        """Shows log for complex types like LocDis, PresContain, Traffic, Residential from their temporary dicts."""
        self.display_anno_log.clear()
        temp_dict_to_show = {}
        if category_key == "LocDis":    temp_dict_to_show = self.dis_loc_details
        elif category_key == "PresContain": temp_dict_to_show = self.contain_details
        elif category_key == "Traffic":     temp_dict_to_show = self.traffic_details
        elif category_key == "Residential": temp_dict_to_show = self.residential_details

        if not temp_dict_to_show:
            self.display_anno_log.setText(f"No '{category_key}' items confirmed yet for this image.")
            return

        log_text = f"Confirmed '{category_key}' items for current image:\n"
        if category_key == "LocDis":
            for (obj_a, obj_b), details in temp_dict_to_show.items():
                log_text += f"  - Pair: ({obj_a}) & ({obj_b}), Loc: {details[1]}, Dist: {details[2]}\n"
        else: # PresContain, Traffic, Residential (similar structure)
            for landcover, items_list in temp_dict_to_show.items():
                log_text += f"  {landcover}:\n"
                for item_attrs in items_list: # item_attrs is a list of [index, text]
                    attr_strings = [f"{self._get_attr_name_for_log(category_key, idx, item_attrs[0])}: {item_attrs[1]}" for idx, item_attrs in enumerate(item_attrs)]
                    log_text += f"    - {', '.join(attr_strings)}\n"
        self.display_anno_log.setText(log_text)


    # --- Image Measurement Triggers ---
    def trigger_distance_measurement(self):
        if self.img_paths_rgb and self.current_image_index < len(self.img_paths_rgb):
            current_image_path = self.img_paths_rgb[self.current_image_index]
            measurement_tool = ImageMeasurement(current_image_path)
            measurement_tool.measure_distance()
        else:
            self._show_warning("Please select an RGB image folder and an image first.")

    def trigger_area_measurement(self):
        if self.img_paths_rgb and self.current_image_index < len(self.img_paths_rgb):
            current_image_path = self.img_paths_rgb[self.current_image_index]
            measurement_tool = ImageMeasurement(current_image_path)
            measurement_tool.measure_area()
        else:
            self._show_warning("Please select an RGB image folder and an image first.")


    # --- Folder Selection and File Handling ---
    def select_save_folder(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        folder_path = dialog.getExistingDirectory(self, "Select Folder to Save Annotations")
        if folder_path:
            self.selected_folder_label_save.setText(folder_path)
            self.save_folder = folder_path
            # Try to load existing attribution file to resume
            attri_file_path = os.path.join(self.save_folder, "image_attributes.json")
            if os.path.exists(attri_file_path):
                try:
                    with open(attri_file_path, "r", encoding='utf-8') as f: # Added encoding
                        loaded_attributes = json.load(f)
                    if loaded_attributes:
                        self.attribution_dict = loaded_attributes
                        if "counter" in self.attribution_dict and self.num_rgb_images > 0 :
                            self.current_image_index = self.attribution_dict["counter"] % self.num_rgb_images
                        else:
                            self.current_image_index = 0
                        self._show_message(f"Resumed from existing annotations. Next image: {self.current_image_index + 1}")
                except json.JSONDecodeError:
                    self._show_warning("Error decoding existing attribution_dict.json. Starting fresh.")
                    self.attribution_dict = {}
                    self.current_image_index = 0
                except Exception as e:
                    self._show_warning(f"Error loading attribution_dict.json: {e}. Starting fresh.")
                    self.attribution_dict = {}
                    self.current_image_index = 0

            else:
                self.current_image_index = 0 # No existing file, start from the beginning
                self.attribution_dict = {} # Ensure it's initialized

            # Update UI if folders are already selected
            if self.selected_folder_rgb and self.selected_folder_tir:
                self._update_image_display_and_attributes()

    def select_rgb_folder(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        folder_path = dialog.getExistingDirectory(self, "Select OPT (Optical) Image Folder")
        if folder_path:
            self.selected_folder_label_rgb.setText(folder_path)
            self.selected_folder_rgb = folder_path
            self.img_paths_rgb, self.rgb_names = get_img_paths(self.selected_folder_rgb)
            self.num_rgb_images = len(self.img_paths_rgb)
            if self.num_rgb_images == 0:
                self._show_warning(f"No images found in OPT folder: {folder_path}")
            # If TIR folder also selected, attempt to display first image
            if self.selected_folder_tir:
                 # Reset counter if save folder isn't set yet or doesn't have resume info
                if not self.save_folder or "counter" not in self.attribution_dict:
                    self.current_image_index = 0
                self._update_image_display_and_attributes()


    def select_tir_folder(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        folder_path = dialog.getExistingDirectory(self, "Select THE (Thermal) Image Folder")
        if folder_path:
            self.selected_folder_label_tir.setText(folder_path)
            self.selected_folder_tir = folder_path
            self.img_paths_tir, self.tir_names = get_img_paths(self.selected_folder_tir)
            self.num_tir_images = len(self.img_paths_tir)
            if self.num_tir_images == 0:
                self._show_warning(f"No images found in THE folder: {folder_path}")
            # If RGB folder also selected, attempt to display first image
            if self.selected_folder_rgb:
                if not self.save_folder or "counter" not in self.attribution_dict:
                    self.current_image_index = 0
                self._update_image_display_and_attributes()


    # --- Image Display and Navigation ---
    def _display_single_image(self, image_path, image_label_widget, panel_width, panel_height):
        """Helper to display an image in a QLabel, scaled to fit."""
        if not image_path or not os.path.exists(image_path):
            image_label_widget.clear()
            image_label_widget.setText("Image not found")
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            image_label_widget.setText("Error loading image")
            return

        scaled_pixmap = pixmap.scaled(panel_width - 20, panel_height - 20, # Margin
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label_widget.setPixmap(scaled_pixmap)

    def _update_image_display(self):
        """Updates the displayed RGB and TIR images and their info labels."""
        if self.num_rgb_images > 0 and self.current_image_index < self.num_rgb_images:
            self._display_single_image(self.img_paths_rgb[self.current_image_index], self.image_box_rgb, self.img_panel_width, self.img_panel_height)
            self.rgb_name_label.setText(f"OPT: {self.rgb_names[self.current_image_index]}")
            self.progress_bar_rgb.setText(f"{self.current_image_index + 1} of {self.num_rgb_images}")
        else:
            self.image_box_rgb.clear()
            self.rgb_name_label.setText("OPT: N/A")
            self.progress_bar_rgb.setText("0 of 0")

        if self.num_tir_images > 0 and self.current_image_index < self.num_tir_images:
             # Assuming TIR images correspond by index if names don't match perfectly
            tir_display_index = self.current_image_index
            # Attempt to find matching TIR image by name (more robust)
            if self.rgb_names and self.tir_names and self.current_image_index < len(self.rgb_names):
                current_rgb_base_name, _ = os.path.splitext(self.rgb_names[self.current_image_index])
                # Adapt this matching logic if your naming convention is different
                # Example: rgb_001.jpg -> tir_001.jpg or optical_001.png -> thermal_001.png
                potential_tir_name_variants = [
                    current_rgb_base_name.replace("rgb", "tir", 1) + ext for ext in [".jpg", ".png", ".jpeg"]
                ] + [
                    current_rgb_base_name.replace("optical", "thermal", 1) + ext for ext in [".jpg", ".png", ".jpeg"]
                ] + [
                     current_rgb_base_name + ext for ext in [".jpg", ".png", ".jpeg"] # if names are identical but in different folders
                ]


                found_match = False
                for tir_idx, tir_name_iter in enumerate(self.tir_names):
                    if tir_name_iter in potential_tir_name_variants or os.path.splitext(tir_name_iter)[0] == current_rgb_base_name : # Simpler match if prefix is same
                        tir_display_index = tir_idx
                        found_match = True
                        break
                if not found_match and self.num_rgb_images == self.num_tir_images:
                    # Fallback to index if no name match but counts are same
                    tir_display_index = self.current_image_index
                elif not found_match:
                    self.image_box_tir.setText("No matching THE image found by name.")
                    self.tir_name_label.setText("THE: N/A")
                    self.progress_bar_tir.setText(f"{self.current_image_index + 1} of {self.num_tir_images} (approx)")
                    return # Don't try to display if no good match

            if tir_display_index < self.num_tir_images:
                self._display_single_image(self.img_paths_tir[tir_display_index], self.image_box_tir, self.img_panel_width, self.img_panel_height)
                self.tir_name_label.setText(f"THE: {self.tir_names[tir_display_index]}")
                self.progress_bar_tir.setText(f"{tir_display_index + 1} of {self.num_tir_images}")
            else: # Should not happen if logic above is correct
                self.image_box_tir.clear()
                self.tir_name_label.setText("THE: Index out of bounds")
                self.progress_bar_tir.setText("Error")

        else:
            self.image_box_tir.clear()
            self.tir_name_label.setText("THE: N/A")
            self.progress_bar_tir.setText("0 of 0")

    def show_next_image(self):
        if not self.selected_folder_rgb or not self.selected_folder_tir or not self.save_folder:
            self._show_warning("Please select RGB, TIR, and Save folders first.")
            return
        if self.num_rgb_images == 0:
             self._show_warning("No images loaded in the OPT folder.")
             return

        # Ensure current annotations are "saved" to memory before moving
        self.save_current_image_annotations_to_memory(show_success_message=False)

        if self.current_image_index < self.num_rgb_images - 1:
            self.current_image_index += 1
            self._update_image_display_and_attributes()
        else:
            self._show_message("This is the last image. Annotations complete or save all to file.")

    def show_previous_image(self):
        if not self.selected_folder_rgb or not self.selected_folder_tir or not self.save_folder:
            self._show_warning("Please select RGB, TIR, and Save folders first.")
            return
        if self.num_rgb_images == 0:
             self._show_warning("No images loaded in the OPT folder.")
             return

        self.save_current_image_annotations_to_memory(show_success_message=False)

        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._update_image_display_and_attributes()
        else:
            self._show_message("This is the first image.")

    def _update_image_display_and_attributes(self):
        """Central function to call when image changes or folders are selected."""
        if not self.img_paths_rgb or self.current_image_index >= len(self.img_paths_rgb):
            # This can happen if folders are selected but then cleared, or index is bad
            self._clear_ui_for_new_image() # Clear everything if no valid image
            return

        self.current_rgb_name = self.rgb_names[self.current_image_index]
        # Try to get corresponding TIR name (logic might need adjustment based on naming)
        # For now, assume TIR name matches RGB name if counts are equal, or use index
        if self.num_rgb_images == self.num_tir_images and self.current_image_index < len(self.tir_names):
            self.current_tir_name = self.tir_names[self.current_image_index]
        elif self.current_image_index < len(self.tir_names): # If TIR has more images, use corresponding index
             self.current_tir_name = self.tir_names[self.current_image_index]
        else:
            self.current_tir_name = "N/A" # Or handle mismatch

        self._update_image_display()
        self._load_annotations_for_current_image() # Load existing annotations
        self._reset_annotation_panel_ui()          # Reset UI elements to reflect loaded or new state
        self._display_current_attributes_in_log()  # Show loaded/current attributes

    def _clear_ui_for_new_image(self):
        """Clears image displays, info labels, and annotation panel."""
        self.image_box_rgb.clear()
        self.image_box_tir.clear()
        self.rgb_name_label.setText("OPT: N/A")
        self.progress_bar_rgb.setText("0 of 0")
        self.tir_name_label.setText("THE: N/A")
        self.progress_bar_tir.setText("0 of 0")
        self.current_image_attributes = {}
        self.dis_loc_details = {}
        self.contain_details = {}
        self.traffic_details = {}
        self.residential_details = {}
        self._reset_annotation_panel_ui()
        self.display_anno_log.clear()


    def _reset_annotation_panel_ui(self):
        """Resets all checkboxes and comboboxes in the annotation panel."""
        self.allow_modification = False # Prevent signals during reset
        for chk_box in self.all_checkboxes:
            chk_box.setChecked(False) # Default to unchecked

        for widget in self.all_comboboxes_and_lineedits:
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(0) # Reset to the first item (placeholder)
            elif isinstance(widget, QLineEdit):
                # Reset to placeholder text if defined, otherwise clear
                if "custom attribute 1 key" in widget.placeholderText() or "Enter custom attribute 1 key" == widget.text():
                     widget.setText("Enter custom attribute 1 key")
                elif "attribute 1 value" in widget.placeholderText() or "Enter attribute 1 value" == widget.text():
                     widget.setText("Enter attribute 1 value")
                elif "custom attribute 2 key" in widget.placeholderText() or "Enter custom attribute 2 key" == widget.text():
                     widget.setText("Enter custom attribute 2 key")
                elif "attribute 2 value" in widget.placeholderText() or "Enter attribute 2 value" == widget.text():
                     widget.setText("Enter attribute 2 value")
                else:
                    widget.clear()

        # Now, load existing attributes and set UI elements accordingly
        if self.current_rgb_name in self.attribution_dict:
            attrs = self.attribution_dict[self.current_rgb_name]
            # Simple attributes
            if "match_condition" in attrs: self.chk_match.setChecked(True); self.match_options_box.setCurrentText(attrs["match_condition"])
            if "mist_condition" in attrs: self.chk_match.setChecked(True); self.mist_options_box.setCurrentText(attrs["mist_condition"])
            if "darkness_condition" in attrs: self.chk_match.setChecked(True); self.night_options_box.setCurrentText(attrs["darkness_condition"])
            if "area_type" in attrs: self.chk_theme.setChecked(True); self.theme_residential_box.setCurrentText(attrs["area_type"])
            if "scene_macro_category" in attrs: self.chk_theme.setChecked(True); self.theme_urban_rural_box.setCurrentText(attrs["scene_macro_category"])
            if "agricultural_road" in attrs: self.chk_agricultural.setChecked(True); self.agri_road_box.setCurrentText(attrs["agricultural_road"])
            if "agricultural_water" in attrs: self.chk_agricultural.setChecked(True); self.agri_water_box.setCurrentText(attrs["agricultural_water"])
            if "industrial_facility" in attrs: self.chk_industrial.setChecked(True); self.ind_facility_box.setCurrentText(attrs["industrial_facility"])
            if "industrial_scale" in attrs: self.chk_industrial.setChecked(True); self.ind_scale_box.setCurrentText(attrs["industrial_scale"])
            if "industrial_location" in attrs: self.chk_industrial.setChecked(True); self.ind_location_box.setCurrentText(attrs["industrial_location"])
            if "uav_height" in attrs: self.chk_uav.setChecked(True); self.uav_height_box.setCurrentText(attrs["uav_height"])
            if "uav_angle" in attrs: self.chk_uav.setChecked(True); self.uav_angle_box.setCurrentText(attrs["uav_angle"])

            # Complex attributes (LocDis, PresContain, Traffic, Residential) - these are loaded into their temporary dicts
            # The UI for these is typically populated when their respective landcover/object boxes are changed by the user.
            # Or, you could try to pre-fill the *first* item of a complex attribute if it exists.
            if "LocDis" in attrs:
                self.chk_dis_loc.setChecked(True)
                self.dis_loc_details = {tuple(item[:2]): item[2:] for item in attrs["LocDis"]} # Rebuild dict
            if "PresContain" in attrs:
                self.chk_contain.setChecked(True)
                self.contain_details = attrs["PresContain"] # This is already a dict
            if "Traffic" in attrs:
                self.chk_traffic.setChecked(True)
                self.traffic_details = attrs["Traffic"]
            if "Residential" in attrs:
                self.chk_residential.setChecked(True)
                self.residential_details = attrs["Residential"]

            # Custom Deduce attributes
            # This requires iterating through attrs to find keys not matching predefined ones
            # Or, if Deduce attributes are stored under a specific key like "CustomAttributes":
            if "CustomAttributes" in attrs and isinstance(attrs["CustomAttributes"], dict):
                self.chk_deduce.setChecked(True)
                custom_attr_list = list(attrs["CustomAttributes].items())
                if len(custom_attr_list) > 0:
                    self.deduce_q1_input.setText(custom_attr_list[0][0])
                    self.deduce_a1_input.setText(str(custom_attr_list[0][1])) # Ensure string
                    self.deduce_q1_input.setProperty("custom_key", custom_attr_list[0][0])
                if len(custom_attr_list) > 1:
                    self.deduce_q2_input.setText(custom_attr_list[1][0])
                    self.deduce_a2_input.setText(str(custom_attr_list[1][1]))
                    self.deduce_q2_input.setProperty("custom_key", custom_attr_list[1][0])


        self.allow_modification = True


    # --- Annotation Saving and Loading ---
    def _load_annotations_for_current_image(self):
        """Loads annotations for the current image from self.attribution_dict."""
        self.current_image_attributes = {} # Clear previous
        self.dis_loc_details = {}
        self.contain_details = {}
        self.traffic_details = {}
        self.residential_details = {}

        if self.current_rgb_name in self.attribution_dict:
            self.current_image_attributes = self.attribution_dict[self.current_rgb_name].copy() # Load a copy
            # If complex attributes are stored directly, load them into their temp dicts
            if "LocDis" in self.current_image_attributes and isinstance(self.current_image_attributes["LocDis"], list):
                self.dis_loc_details = {tuple(item[:2]): item[2:] for item in self.current_image_attributes["LocDis"]}
            if "PresContain" in self.current_image_attributes and isinstance(self.current_image_attributes["PresContain"], dict):
                self.contain_details = self.current_image_attributes["PresContain"].copy()
            if "Traffic" in self.current_image_attributes and isinstance(self.current_image_attributes["Traffic"], dict):
                self.traffic_details = self.current_image_attributes["Traffic"].copy()
            if "Residential" in self.current_image_attributes and isinstance(self.current_image_attributes["Residential"], dict):
                self.residential_details = self.current_image_attributes["Residential"].copy()
        # The UI will be updated by _reset_annotation_panel_ui after this

    def save_current_image_annotations_to_memory(self, show_success_message=True):
        """Saves the current_image_attributes to the main self.attribution_dict."""
        if not self.current_rgb_name:
            if show_success_message: # Only show warning if user explicitly clicked save
                 self._show_warning("No current image selected to save annotations for.")
            return

        # Finalize any pending complex attributes before saving
        if self.chk_dis_loc.isChecked() and self.dis_loc_details : self._finalize_disloc_attributes()
        if self.chk_contain.isChecked() and self.contain_details : self._finalize_contain_attributes()
        if self.chk_traffic.isChecked() and self.traffic_details : self._finalize_traffic_attributes()
        if self.chk_residential.isChecked() and self.residential_details: self._finalize_residential_attributes()


        if self.current_image_attributes: # Only save if there are attributes
            self.attribution_dict[self.current_rgb_name] = self.current_image_attributes.copy() # Save a copy
            if show_success_message:
                self._show_message(f"Attributes for '{self.current_rgb_name}' saved to memory.")
        elif self.current_rgb_name in self.attribution_dict: # If no current attributes but was previously saved, remove it
            del self.attribution_dict[self.current_rgb_name]
            if show_success_message:
                self._show_message(f"Cleared attributes for '{self.current_rgb_name}' from memory.")


    def save_all_annotations_to_file(self):
        """Saves the entire self.attribution_dict to a JSON file."""
        if not self.save_folder:
            self._show_warning("Please select a save folder first.")
            return
        if not self.attribution_dict:
            self._show_message("No annotations to save to file.")
            return

        # Before saving, ensure the current image's annotations are in memory
        self.save_current_image_annotations_to_memory(show_success_message=False)

        # Add/update the counter for resuming
        self.attribution_dict["counter"] = self.current_image_index

        file_path = os.path.join(self.save_folder, "image_attributes.json")
        try:
            with open(file_path, "w", encoding='utf-8') as f: # Added encoding
                json.dump(self.attribution_dict, f, indent=4, sort_keys=True)
            self._show_message(f"All annotations successfully saved to:\n{file_path}")
        except Exception as e:
            self._show_warning(f"Error saving annotations to file: {e}")


    # --- UI Helper Methods (Warnings, Messages) ---
    def _show_warning(self, message):
        self.display_anno_log.clear()
        self.display_anno_log.setText(f"WARNING: {message}")
        # QtWidgets.QMessageBox.warning(self, "Warning", message) # Alternative: use QMessageBox

    def _show_message(self, message):
        self.display_anno_log.clear()
        self.display_anno_log.setText(f"INFO: {message}")
        # QtWidgets.QMessageBox.information(self, "Information", message)

    def closeEvent(self, event):
        """Handles the window close event to auto-save annotations."""
        reply = QtWidgets.QMessageBox.question(self, 'Confirm Exit',
                                           "Save all annotations before exiting?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
                                           QtWidgets.QMessageBox.Save)
        if reply == QtWidgets.QMessageBox.Save:
            self.save_all_annotations_to_file()
            event.accept()
        elif reply == QtWidgets.QMessageBox.Discard:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    annotation_tool = AnnotationWindow()
    annotation_tool.show()
    sys.exit(app.exec_())
