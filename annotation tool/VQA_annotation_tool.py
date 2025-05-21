# Version --- Added optical-thermal infrared questions

import json
import sys
import os

import random
import math
import numpy as np
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QMimeDatabase
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
)


def coss_multi(v1, v2):

    return v1[0] * v2[1] - v1[1] * v2[0]


def polygon_area(polygon):

    n = len(polygon)
    if n < 3:
        return 0
    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]
    area = 0
    for i in range(1, n):
        area = area + coss_multi(vectors[i - 1, :], vectors[i, :]) / 2
    return area


class img:
    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(self.path)
        self.coordinate_d = []
        self.coordinate_a = []
        self.count_a = 0
        self.count_d = 0

    def event_d(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinate_d.append([x, y])
            cv2.circle(self.img, (x, y), 1, (0, 255, 0), thickness=-1)
            cv2.imshow("Distance", self.img)

        if event == cv2.EVENT_MBUTTONDOWN:
            self.count_d = self.count_d + 1
            if len(self.coordinate_d) >= 2:
                sm = math.pow(
                    (self.coordinate_d[-1][0] - self.coordinate_d[-2][0]), 2
                ) + math.pow((self.coordinate_d[-1][1] - self.coordinate_d[-2][1]), 2)
                distance = math.pow(sm, 0.5)
            else:
                distance = 0
            cv2.putText(
                self.img,
                "Distance_%d:%d" % (self.count_d, distance),
                (120, 70 + self.count_d * 20),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 255, 230),
                thickness=1,
            )
            cv2.imshow("Distance", self.img)

    def get_dis(self):
        self.img = cv2.imread(self.path)
        cv2.namedWindow("Distance", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Distance", self.event_d)
        cv2.imshow("Distance", self.img)
        cv2.waitKey(0)

    def event_a(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.coordinate_a.append([x, y])

            cv2.circle(self.img, (x, y), 1, (0, 255, 0), thickness=-1)
            # cv2.putText(self.img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,0.9, (0, 255, 0), thickness=1)
            cv2.imshow("Area", self.img)

        if event == cv2.EVENT_MBUTTONDOWN:
            self.count_a = self.count_a + 1
            polygon = np.array(self.coordinate_a)
            Area = -1 * polygon_area(polygon)
            cv2.putText(
                self.img,
                "Area_%d:%d" % (self.count_a, Area),
                (120, 70 + self.count_a * 20),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 255, 230),
                thickness=1,
            )

            cv2.imshow("Area", self.img)
            self.coordinate_a = []

    def get_area(self):
        self.img = cv2.imread(self.path)
        cv2.namedWindow("Area", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Area", self.event_a)
        cv2.imshow("Area", self.img)
        cv2.waitKey(0)


def get_img_paths(dir, extensions=(".jpg", ".png", ".jpeg")):
    """
    :param dir: folder with files
    :param extensions: tuple with file endings. e.g. ('.jpg', '.png'). Files with these endings will be added to img_paths
    :return: list of all filenames
    """
    img_paths = []
    img_name = []

    for filename in os.listdir(dir):
        if filename.lower().endswith(extensions):
            img_paths.append(os.path.join(dir, filename))
            img_name.append(filename)

    return img_paths, img_name


class Annotation_window(QWidget):
    def __init__(self):
        super().__init__()

        # Define variables to store important data
        # Define folder path storage locations
        self.selected_folder_rgb = ""
        self.selected_folder_tir = ""
        self.save_folder = ""  # Folder to store annotation files

        # Absolute paths of image files
        self.img_paths_rgb = []
        self.img_paths_tir = []
        # List of image file names
        self.rgb_name = []
        self.tir_name = []
        self.num_rgb = 0  # Total number of images in folder
        self.num_tir = 0
        self.counter = 0  # Current image index

        # Store total data, initialize when loading annotation file
        # Store total image annotation data {img_name:{q_id:[qst,ans],q_id:[qst,ans],...},tir2:{q_id:[qst,ans],q_id:[qst,ans],...},...}
        self.annotation_dict = {}

        self.attribution_dict = ({})  # Store total image attributes {img_name:{attribute1:,attribute2:,...}}

        # Must update when changing images, initialize when changing images
        # Store current image annotation questions {img_name:[qst,ans],q_id:[qst,ans],...}
        self.current_qst_dict = {}

        self.current_attri_dict = {}  # Store current attributes {attribute1:,attribute2:,...}

        self.current_tir_name = ""  # Store current image name
        self.current_rgb_name = ""

        self.flag = True  # When flag is true, annotation content can be modified

        # When modifying landcover, only delete current landcover, not all landcovers
        self.current_landcover = "None"

        self.current_traffic_landcover_subset = "None"
        self.current_residential_landcover_subset = "None"

        # Dictionary
        # self.object_dict=['dog','cat','plane','house','twon'] # Object dictionary

        # Set window size and display widgets
        # Window size and title
        self.title = "VQA (Visual Question Answering) Image Annotation Tool"
        self.left = 200
        self.top = 100
        self.width = 1580
        self.height = 770
        # Image display area size
        self.img_panel_width = 450
        self.img_panel_height = 450

        # Residential area problem answers
        self.residential_ans_landcover = QtWidgets.QComboBox(self)
        self.residential_ans_object = QtWidgets.QComboBox(self)
        self.residential_ans_number = QtWidgets.QComboBox(self)
        self.residential_ans_location = QtWidgets.QComboBox(self)
        self.residential_ans_better = QtWidgets.QComboBox(self)
        self.submit_residential = QtWidgets.QPushButton("Confirm", self)
        self.residential_list = [
            self.residential_ans_landcover,
            self.residential_ans_object,
            self.residential_ans_number,
            self.residential_ans_location,
            self.residential_ans_better,
        ]
        self.residential_landcover_list = [
            "living environment",
            "construction type",
        ]
        self.residential_object_list = [
            # Living environment
            "recreational area",
            "commercial area",
            "construction area",
            "river",
            "lake",
            "linear walkway",
            "curved walkway",
            "no visible walkway",
            # Buildings
            "low-rise residential building",
            "high-rise residential building",
            "low-rise non-residential building",
            "high-rise non-residential building",
        ]
        self.RA2ObjList = {
            "living environment": [
                "Object",
                "recreational area",
                "commercial area",
                "construction area",
                "river",
                "lake",
                "linear walkway",
                "curved walkway",
                "no visible walkway",
            ],
            "construction type": [
                "Object",
                "low-rise residential building",
                "high-rise residential building",
                "low-rise non-residential building",
                "high-rise non-residential building",
            ],
            "LandCover Class": ["Object Class"],
        }

        #  self.RA2ObjList = {
        #     "living environment": ["recreational area","commercial area","construction area","river","lake","linear walkway","curved walkway","no visible walkway"],
        #     "construction type": ["low-rise residential building","high-rise residential building","low-rise non-residential building","high-rise non-residential building"]}

        # Traffic problem answers
        self.traffic_ans_landcover = QtWidgets.QComboBox(self)
        self.traffic_ans_object = QtWidgets.QComboBox(self)
        self.traffic_ans_number = QtWidgets.QComboBox(self)
        self.traffic_ans_location = QtWidgets.QComboBox(self)
        self.traffic_ans_better = QtWidgets.QComboBox(self)
        self.submit_traffic = QtWidgets.QPushButton("Confirm", self)
        self.traffic_list = [
            self.traffic_ans_landcover,
            self.traffic_ans_object,
            self.traffic_ans_number,
            self.traffic_ans_location,
            self.traffic_ans_better,
        ]
        self.traffic_landcover_list = [
            "road type",
            "vehical",
            "pedestrian",
            "road facility",
            "road condition",
            "vehicle traffic violation",
            "pedestrian traffic violation",
            "vehicle behavior",
            "pedestrian behavior",
        ]
        self.traffic_landcover_list = [
            "road type",
            "vehical",
            "pedestrian",
            "road facility",
            "road condition",
            "vehicle traffic violation",
            "non-motor vehicle violation",
            "pedestrian traffic violation",
            "vehicle behavior",
            "non-motor vehicle behavior",
            "pedestrian behavior",
            "abnormal traffic situation",
            "traffic participant interaction"
        ]
        self.traffic_object_list = [
            # Road types
            "main city road",
            "street",
            "quick road",
            "residential street",
            "alley",
            "intersection",
            "lane merge",
            "pedestrian crossing",
            "bridge",
            "non-motorized road",
            "unpaved road",
            "bus lane",
            "gridline",
            "overhead walkway",
            "other paved road",
            # Vehicle categories
            "car",
            "large vehicle",
            "other vehicle",
            # Pedestrian categories
            "single pedestrian",
            "pedestrian group",
            # Road facilities
            "motor vehicle parking spot",
            "non-motorised parking spot",
            "road divider",
            # Road conditions
            "normal pavement",
            "damaged pavement",
            "road construction",
            # Road conditions
            "illegal parking",
            "go against one-way traffic",
            "Illegal lane change",
            "run the red light",
            "vehicle on solid line",
            # Non-motorized vehicle traffic violations
            "illegal passenger carrying",
            "wrong-way driving",
            "running red light",
            "improper lane usage",
            "improper parking",
            "no safety helmet",
            # Pedestrian traffic violations
            "failure to use crosswalks",
            "walking on non-sidewalks",
            "run the red light",
            "other violations",
            # Vehicle behaviors
            "lane change",
            "vehicle  turn",
            "vehicle U-turn",
            "overtake",
            "vehicle queuing",
            "traffic congestion",
            "too close to another car",
            # Non-motorized vehicle behaviors
            "waiting at traffic light",
            "normal driving in non-motor lane",
            # Pedestrian behaviors
            "wait for a traffic light",
            "walk on the crosswalk",
            "walk on the sidewalk",
            # Abnormal traffic situations
            "traffic accident",
            "traffic jam",
            # Traffic participant interaction behaviors
            "vehicle yielding to pedestrian",
            "vehicle waiting for boarding",
            "vehicle waiting for alighting",
            "bus temporary stop",
            "vehicle entering parking lot",
            "vehicle exiting parking lot"
        ]
        self.TR2ObjList = {
            "road type": [
                "Object",
                "main city road",
                "street",
                "quick road",
                "residential street",
                "alley",
                "intersection",
                "lane merge",
                "pedestrian crossing",
                "bridge",
                "non-motorized road",
                "unpaved road",
                "bus lane",
                "gridline",
                "overhead walkway",
                "other paved road",
            ],
            "vehical": [
                "Object",
                "car",
                "large vehicle",
                "other vehicle",
            ],
            "pedestrian": ["Object", "single pedestrian", "pedestrian group"],
            "road facility": [
                "Object",
                "motor vehicle parking spot",
                "non-motorised parking spot",
                "lane marking",
                "road divider",
            ],
            "road condition": [
                "Object",
                "normal pavement",
                "damaged pavement",
                "road construction",
            ],
            "vehicle traffic violation": [
                "Object",
                "illegal parking",
                "go against one-way traffic",
                "Illegal lane change",
                "run the red light",
                "vehicle on solid line",
            ],
            "non-motor vehicle violation": [
                "Object",
                "illegal passenger carrying",
                "driving in the wrong direction",
                "run the red light",
                "improper lane usage",
                "improper parking",
                "no safety helmet"
            ],
            "pedestrian traffic violation": [
                "Object",
                "failure to use crosswalks",
                "walking on non-sidewalks",
                "run the red light",
                "other violations",
            ],
            "vehicle behavior": [
                "Object",
                "lane change",
                "vehicle  turn",
                "vehicle U-turn",
                "overtake",
                "vehicle queuing",
                "traffic congestion",
                "too close to another car",
            ],
            "non-motor vehicle behavior": [
                "Object",
                "wait for a traffic light",
                "driving in non-motor lane"
            ],
            "pedestrian behavior": [
                "Object",
                "wait for a traffic light",
                "walk on the crosswalk",
                "walk on the sidewalk",
            ],
            "abnormal traffic situation": [
                "Object",
                "traffic accident",
                "traffic jam",
            ],
            "traffic participant interaction": [
                "Object",
                "vehicle yielding to pedestrians",
                "vehicle waiting for passengers to board",
                "vehicle waiting for passengers to alight",
                "bus temporary stop",
                "vehicle entering the parking lot",
                "vehicle exiting the parking lot"
            ],
            "LandCover Class": ["Object Class"],
        }

        # self.TR2ObjList = {
        #     "road type": ["main city road","quick road","residential street","alley","intersection","lane merge","pedestrian crossing","bridge","non-motorized road","unpaved road","bus lane","gridline","overhead walkway","street","other paved road"],
        #     "vehical": ["car","large vehicle","other vehicle",],
        #     "pedestrian": [ "single pedestrian", "pedestrian group"],
        #     "road facility": ["motor vehicle parking spot","non-motorised parking spot","lane marking","road divider"],
        #     "road condition": ["normal pavement","damaged pavement","road construction"],
        #     "vehicle traffic violation": ["illegal parking","go against one-way traffic","Illegal lane change","run the red light","vehicle on solid line"],
        #     "pedestrian traffic violation": ["failure to use crosswalks","walking on non-sidewalks","run the red light","other violations",],
        #     "vehicle behavior": ["lane change","vehicle  turn","vehicle U-turn","overtake","vehicle queuing","traffic congestion","too close to another car",],
        #     "pedestrian behavior": ["wait for a traffic light","walk on the crosswalk","walk on the sidewalk"]}

        # Contain/presence problem answers
        self.contain_ans_landcover = QtWidgets.QComboBox(self)
        self.contain_ans_subset = QtWidgets.QComboBox(self)
        self.contain_ans_number = QtWidgets.QComboBox(self)
        self.contain_ans_location = QtWidgets.QComboBox(self)
        self.contain_ans_shape = QtWidgets.QComboBox(self)
        self.contain_ans_Area = QtWidgets.QComboBox(self)
        self.contain_ans_length = QtWidgets.QComboBox(self)
        self.contain_ans_distribution = QtWidgets.QComboBox(self)
        self.contain_ans_better = QtWidgets.QComboBox(self)
        self.submit_contain = QtWidgets.QPushButton("Confirm", self)

        self.contain_list = [
            self.contain_ans_landcover,
            self.contain_ans_subset,
            self.contain_ans_number,
            self.contain_ans_location,
            self.contain_ans_shape,
            self.contain_ans_Area,
            self.contain_ans_length,
            self.contain_ans_distribution,
            self.contain_ans_better,
        ]
        self.landcover_list = [
            "agricultural area",
            "airport",
            "apron",
            "building",
            "beach",
            "pier",
            "intersection",
            "park",
            "parking area",
            "sports field",
            "road",
            "concrete floor",
            "vegetation area",
            "wasteland",
            "water area",
        ]  # Store all landcover types, including some unseen categories to balance Yes-type questions in presence

        self.subset_list = [
            "woodland",
            "grassland",
            "other vegetation area",
            "pond",
            "river",
            "ditch",
            "sea",
            "lake",
            "other water area",
            "wide road",
            "narrow road",
            "low-rise residential building",
            "high-rise residential building",
            "low-rise non-residential building",
            "high-rise non-residential building",
            "agricultural area",
            "wasteland",  # barren land
            "intersection",
            "parking area",
            "park",
            "concrete floor",  # concrete floor
            # Sports field 
            "basketball court",
            "football field",
            "baseball field",
            "athletic track",
            "tennis courts",
            "pier",  # pier 
            "beach",
            "airport",
            "apron",
        ]  # Store all subset types, including some unseen categories to balance Yes-type questions in presence

        # self.subset_list=sorted(sub_list,key=lambda word:word[:2]) # Sort by first two letters

        self.LC2SubList = {
            "building": [
                "Subset",
                "low-rise residential building",
                "high-rise residential building",
                "low-rise non-residential building",
                "high-rise non-residential building",
            ],
            "vegetation area": [
                "Subset",
                "woodland",
                "grassland",
                "other vegetation area",
            ],
            "water area": [
                "Subset",
                "ditch",
                "pond",
                "river",
                "sea",
                "lake",
                "other water area",
            ],
            "road": ["Subset", "wide road", "narrow road"],
            "agricultural area": ["Subset", "agricultural area"],
            "wasteland": ["Subset", "wasteland"],
            "intersection": ["Subset", "intersection"],
            "parking area": ["Subset", "parking area"],
            "park": ["Subset", "park"],
            "concrete floor": ["Subset", "concrete floor"],
            "sports field": [
                "Subset",
                "basketball court",
                "baseball field",
                "football field",
                "tennis courts",
                "athletic track",
            ],
            "pier": ["Subset", "pier"],
            "beach": ["Subset", "beach"],
            "airport": ["Subset", "airport"],
            "apron": ["Subset", "apron"],
            "LandCover Class": ["Subset Class"],
        }

        # self.LC2SubList = {
        #     "building": ["low-rise residential building","high-rise residential building","low-rise non-residential building","high-rise non-residential building",],
        #     "vegetation area": ["woodland","grassland","other vegetation area",],
        #     "water area": ["ditch","pond","river","sea","lake","other water area",],
        #     "road": [ "wide road", "narrow road"],
        #     "agricultural area": [ "agricultural area"],
        #     "wasteland": [ "wasteland"],
        #     "intersection": [ "intersection"],
        #     "parking area": [ "parking area"],
        #     "park": [ "park"],
        #     "concrete floor": [ "concrete floor"],
        #     "sports field": ["basketball court","baseball field","football field","tennis courts","athletic track",],
        #     "pier": [ "pier"],
        #     "beach": [ "beach"],
        #     "airport": [ "airport"],
        #     "apron": [ "apron"]}
        #  , 'water area'
        # Store distance and location information for different objects {(A,B):[dis_index,location,distance]}
        self.dis_loc_dict = {}
        # Store contain/presence collected information {landcover1:[[index,subset],[index,number],...],landcover2:[[index,subset],[index,number],...],...}
        self.contain_dict = {}
        self.traffic_dict = {}
        self.residential_dict = {}

        self.location = [
            "above",
            "below",
            "left",
            "right",
            "upper left",
            "upper right",
            "bottom left",
            "bottom right",
        ]
        self.ablocation = {
            "above": "below",
            "below": "above",
            "left": "right",
            "right": "left",
            "upper left": "bottom right",
            "upper right": "bottom left",
            "bottom left": "upper right",
            "bottom right": "upper left",
        }
        # Define QLabel
        # Set titles
        self.headline_folder_rgb = QLabel("1.select OPT image folder:", self)
        self.headline_folder_tir = QLabel("2.select TIR image folder: ", self)
        self.headline_folder_save = QLabel("3.select save folder: ", self)

        # Display output paths
        self.selected_folder_label_rgb = QLabel(self)
        self.selected_folder_label_tir = QLabel(self)
        self.selected_folder_label_save = QLabel(self)

        # Define browse buttons
        self.browse_button_rgb = QtWidgets.QPushButton("Browse OPT", self)
        self.browse_button_tir = QtWidgets.QPushButton("Browse TIR", self)
        self.browse_button_save = QtWidgets.QPushButton("Browse SAVE", self)

        # Display annotated information
        self.display_anno = QTextEdit(self)

        # Image display area
        self.image_box_rgb = QLabel(self)  # Display RGB image
        self.image_box_tir = QLabel(self)  # Display SAR image

        # Display image information
        self.rgb_name_label = QLabel(self)  # Image name
        self.progress_bar_rgb = QLabel(self)  # Display progress
        self.tir_name_label = QLabel(self)  # Image name
        self.progress_bar_tir = QLabel(self)  # Display progress

        # Annotation information title
        # Question title
        self.question_headline = QLabel(
            "select the questions you want to answer and provide answers", self
        )

        # Question types, checkboxes
        self.question_Match = QCheckBox("Match", self)
        self.question_theme = QCheckBox("Theme", self)
        self.question_dis_loc = QCheckBox("Distance/Location", self)
        self.question_contain = QCheckBox("Contain/Presence", self)
        self.question_deduce = QCheckBox("Deduce", self)
        self.question_traffic = QCheckBox("Traffic", self)
        self.question_residential = QCheckBox("Residential", self)
        self.question_agricultural = QCheckBox("Agricultural", self)
        self.question_industrial = QCheckBox("Industrial", self)
        self.question_uav = QCheckBox("UAV", self)

        # Custom question input boxes
        self.Deduce_Qst_one = QLineEdit("Enter your question 1 (one-word answer)",self)
        self.Deduce_Ans_one = QLineEdit("Enter answer",self)
        self.Deduce_Qst_two = QLineEdit("Enter your question 2 (one-word answer)",self)
        self.Deduce_Ans_two = QLineEdit("Enter answer",self)

        # Various answer checkboxes
        # Match answer
        self.Match_pick_box = QComboBox(self)
        # Whether there is fog
        self.Mist_pick_box = QComboBox(self)
        # Whether it's night
        self.night_pick_box = QComboBox(self)
        # Theme answers
        self.theme_ans_r = QComboBox(self)
        self.theme_ans_ur = QComboBox(self)
        # Distance and location answers
        self.dis_loc_ans_loc_a = QComboBox(self)
        self.dis_loc_ans_object_a = QComboBox(self)
        self.dis_loc_is_cluster_a = QComboBox(self)
        self.dis_loc_is_cluster_b = QComboBox(self)
        self.dis_loc_ans_loc_b = QComboBox(self)
        self.dis_loc_ans_object_b = QComboBox(self)
        self.dis_loc_ans_distance = QComboBox(self)
        self.dis_loc_ans_location = QComboBox(self)

        # UAV answers
        self.uav_ans_height = QComboBox(self)
        self.uav_ans_angle = QComboBox(self)
        # Agricultural area answers
        self.agricultural_ans_road = QComboBox(self)
        self.agricultural_ans_water = QComboBox(self)
        # Industrial area answers
        self.industrial_ans_facility = QComboBox(self)
        self.industrial_ans_scale = QComboBox(self)
        self.industrial_ans_location = QComboBox(self)

        # Define various button controls
        self.Pre_btn = QtWidgets.QPushButton("Pre", self)
        self.Next_btn = QtWidgets.QPushButton("Next", self)
        self.Sub_btn = QtWidgets.QPushButton("Submit", self)
        self.Gen_btn = QtWidgets.QPushButton("Generate", self)

        # Define answer and question library lists
        self.ans_list = [
            self.Match_pick_box,
            self.Mist_pick_box,
            self.night_pick_box,
            self.theme_ans_r,
            self.theme_ans_ur,
            self.dis_loc_ans_loc_a,
            self.dis_loc_is_cluster_a,
            self.dis_loc_ans_object_a,
            self.dis_loc_ans_loc_b,
            self.dis_loc_is_cluster_b,
            self.dis_loc_ans_object_b,
            self.dis_loc_ans_distance,
            self.dis_loc_ans_location,
            self.contain_ans_landcover,
            self.contain_ans_subset,
            self.contain_ans_number,
            self.contain_ans_location,
            self.contain_ans_shape,
            self.contain_ans_Area,
            self.contain_ans_length,
            self.contain_ans_distribution,
            self.contain_ans_better,
            self.Deduce_Qst_one,
            self.Deduce_Ans_one,
            self.Deduce_Qst_two,
            self.Deduce_Ans_two,
            self.traffic_ans_landcover,
            self.traffic_ans_object,
            self.traffic_ans_number,
            self.traffic_ans_location,
            self.traffic_ans_better,
            self.residential_ans_landcover,
            self.residential_ans_object,
            self.residential_ans_number,
            self.residential_ans_location,
            self.residential_ans_better,
            self.agricultural_ans_road,
            self.agricultural_ans_water,
            self.industrial_ans_facility,
            self.industrial_ans_scale,
            self.industrial_ans_location,
            self.uav_ans_height,
            self.uav_ans_angle,
        ]
        self.question_list = [
            self.question_Match,
            self.question_theme,
            self.question_dis_loc,
            self.question_contain,
            self.question_deduce,
            self.question_traffic,
            self.question_residential,
            self.question_agricultural,
            self.question_industrial,
            self.question_uav,
        ]

        self.init_ui()  # Set specific window details in init_ui()

    def init_ui(self):
        # Annotation window title and minimum size
        self.setObjectName("mainwindow")
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)

        # File selection box position and format
        # RGB part control settings
        # Folder selection prompt
        self.headline_folder_rgb.setGeometry(20, 20, 250, 20)
        # self.headline_folder_rgb.setStyleSheet('font-size:18px,font-weight:bold')
        self.headline_folder_rgb.setObjectName("headline")
        # Display folder
        self.selected_folder_label_rgb.setGeometry(280, 20, 500, 23)
        self.selected_folder_label_rgb.setObjectName("selectedFolderLabel")
        # Browse button control position and format
        self.browse_button_rgb.setGeometry(790, 20, 85, 23)
        self.browse_button_rgb.setStyleSheet("font-size:14px")
        self.browse_button_rgb.clicked.connect(self.pick_new_rgb)

        # RGB part control settings
        # Folder selection prompt
        self.headline_folder_tir.setGeometry(20, 50, 250, 20)
        self.headline_folder_tir.setObjectName("headline")
        # Display file folder path
        self.selected_folder_label_tir.setGeometry(280, 50, 500, 23)
        self.selected_folder_label_tir.setObjectName("selectedFolderLabel")
        # Browse rgb button
        self.browse_button_tir.setGeometry(790, 50, 85, 23)
        self.browse_button_tir.setStyleSheet("font-size:14px")
        self.browse_button_tir.clicked.connect(self.pick_new_tir)

        # Save annotated file folder settings
        # Select file prompt title
        self.headline_folder_save.setGeometry(20, 80, 250, 20)
        self.headline_folder_save.setObjectName("headline")
        # Display stored file path
        self.selected_folder_label_save.setGeometry(280, 80, 500, 23)
        self.selected_folder_label_save.setObjectName("selectedFolderLabel")
        # Set browse save button
        self.browse_button_save.setGeometry(790, 80, 85, 23)
        self.browse_button_save.setStyleSheet("font-size:14px")
        self.browse_button_save.clicked.connect(self.pick_save)
        # Image display in display function

        # Display annotated question generation status
        self.display_anno.setGeometry(20, 590, 885, 160)

            # Question design part
        X_base = self.img_panel_width * 2 + 15  # Start x coordinate of question part

        # Add a scroll bar
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setGeometry(X_base, 20, 600, 700)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.scroll_content)

        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)

        # Set question title
        self.question_headline.setGeometry(X_base + 10, 20, 500, 20)
        self.question_headline.setObjectName("headline")

        # Set question checkboxes and get answers
        # match question
        self.question_Match.setChecked(False)
        self.question_Match.setGeometry(X_base + 37, 60, 260, 20)
        self.question_Match.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_Match.stateChanged.connect(self.ans_match)

        # theme question
        self.question_theme.setChecked(False)
        self.question_theme.setGeometry(X_base + 37, 100, 260, 20)
        self.question_theme.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_theme.stateChanged.connect(self.ans_theme)
        # distance/location question
        self.question_dis_loc.setChecked(False)
        self.question_dis_loc.setGeometry(X_base + 37, 140, 260, 20)
        self.question_dis_loc.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_dis_loc.stateChanged.connect(self.del_dis_loc)

        # self.question_dis_loc.stateChanged.connect(self.get_object_one)
        # self.question_dis_loc.stateChanged.connect(self.a)
        # contain/presence question
        self.question_contain.setChecked(False)
        self.question_contain.setGeometry(X_base + 37, 300, 260, 20)
        self.question_contain.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_contain.stateChanged.connect(self.del_contain)
        # Updated on February 19, 2024, for prescontain, only one landcover will be deleted each time instead of all
        # self.question_contain.stateChanged.connect(self.del_pres_contain_attribution)

        # Deduce question
        self.question_deduce.setChecked(False)
        self.question_deduce.setGeometry(X_base + 37, 470, 350, 20)
        self.question_deduce.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_deduce.stateChanged.connect(self.ans_deduce_one)
        self.question_deduce.stateChanged.connect(self.ans_deduce_two)
        self.question_deduce.stateChanged.connect(self.del_deduce)

        # Traffic question
        self.question_traffic.setChecked(False)
        self.question_traffic.setGeometry(X_base + 37, 340, 260, 20)
        self.question_traffic.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_traffic.stateChanged.connect(self.del_traffic)

        # Residential question
        self.question_residential.setChecked(False)
        self.question_residential.setGeometry(X_base + 37, 380, 260, 20)
        self.question_residential.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_residential.stateChanged.connect(self.del_residential)

        # Agricultural question
        self.question_agricultural.setChecked(False)
        self.question_agricultural.setGeometry(X_base + 37, 380, 260, 20)
        self.question_agricultural.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_agricultural.stateChanged.connect(self.ans_agricultural)

        # Industrial question
        self.question_industrial.setChecked(False)
        self.question_industrial.setGeometry(X_base + 37, 420, 260, 20)
        self.question_industrial.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_industrial.stateChanged.connect(self.ans_industrial)

        # UAV question
        self.question_uav.setChecked(False)
        self.question_uav.setGeometry(X_base + 37, 460, 260, 20)
        self.question_uav.setStyleSheet(
            "font-size: 17px;font-family:SimHei;font-weight:bold"
        )
        self.question_uav.stateChanged.connect(self.ans_uav)

        # Checkbox answer
        # match question answer
        # self.Match_pick_box.addItems(['请选择','Yes','No'])
        self.Match_pick_box.addItems(
            ["Select", "almost match", "partial match", "not match"]
        ) 
        self.Match_pick_box.setGeometry(X_base + 167, 60, 120, 23)
        self.Match_pick_box.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.Match_pick_box.currentIndexChanged.connect(self.ans_match)

        self.Mist_pick_box.addItems(
            ["Check for mist", "mist", "not mist", "not sure"]
        )  
        self.Mist_pick_box.setGeometry(X_base + 297, 60, 120, 23)
        self.Mist_pick_box.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.Mist_pick_box.currentIndexChanged.connect(self.ans_mist)

        self.night_pick_box.addItems(
            ["Check for night", "dark", "not dark", "not sure"]
        )  
        self.night_pick_box.setGeometry(X_base + 427, 60, 120, 23)
        self.night_pick_box.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.night_pick_box.currentIndexChanged.connect(self.ans_night)

        # theme(residential or not)
        self.theme_ans_r.addItems(["Residential Area", "Residential", "n-Residential"])
        self.theme_ans_r.setGeometry(X_base + 167, 100, 120, 23)
        self.theme_ans_r.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.theme_ans_r.currentIndexChanged.connect(self.ans_theme)

        # ubran or rural
        self.theme_ans_ur.addItems(["Urban/Rural", "Urban", "Rural"])
        self.theme_ans_ur.setGeometry(X_base + 297, 100, 120, 23)
        self.theme_ans_ur.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.theme_ans_ur.currentIndexChanged.connect(self.ans_theme)

        # agricultural road question
        self.agricultural_ans_road.addItems(["Agricultural Road", "Yes", "No"])
        self.agricultural_ans_road.setGeometry(X_base + 167, 380, 120, 23)
        self.agricultural_ans_road.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.agricultural_ans_road.currentIndexChanged.connect(self.ans_agricultural)

        # agricultural water question
        self.agricultural_ans_water.addItems(["Agricultural Water", "Yes", "No"])
        self.agricultural_ans_water.setGeometry(X_base + 297, 380, 120, 23)
        self.agricultural_ans_water.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.agricultural_ans_water.currentIndexChanged.connect(self.ans_agricultural)

        # industrial facility question
        self.industrial_ans_facility.addItems(["Industrial Facility", "Yes", "No"])
        self.industrial_ans_facility.setGeometry(X_base + 167, 420, 120, 23)
        self.industrial_ans_facility.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.industrial_ans_facility.currentIndexChanged.connect(self.ans_industrial)

        # industrial scale question
        self.industrial_ans_scale.addItems(["Industrial Scale", "small", "medium", "large"])
        self.industrial_ans_scale.setGeometry(X_base + 297, 420, 120, 23)
        self.industrial_ans_scale.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.industrial_ans_scale.currentIndexChanged.connect(self.ans_industrial)

        # industrial location question
        self.industrial_ans_location.addItems(
            [
                "Industrial Area Location",
                "top",
                "bottom",
                "left",
                "right",
                "center",
                "top left",
                "top right",
                "bottom left",
                "bottom right",
            ]
        )
        self.industrial_ans_location.setGeometry(X_base + 327, 460, 120, 23)
        self.industrial_ans_location.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.industrial_ans_location.currentIndexChanged.connect(self.ans_industrial)

        # uav height question
        self.uav_ans_height.addItems(["Height", "150-250", "250-400", "400-550", "none"])
        self.uav_ans_height.setGeometry(X_base + 167, 460, 120, 23)
        self.uav_ans_height.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.uav_ans_height.currentIndexChanged.connect(self.ans_uav)

        # uav angle question
        self.uav_ans_angle.addItems(["Angle", "vertical", "oblique"])
        self.uav_ans_angle.setGeometry(X_base + 297, 460, 120, 23)
        self.uav_ans_angle.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.uav_ans_angle.currentIndexChanged.connect(self.ans_uav)

        # distance and location question answer
        # Title
        # ['LocationA','top','bottom','left','right','central','upper left','upper right','lower left','lower right']
        # ['LocationA','above','below','left','right','upper left','upper right','lower left','lower right']
        # distance/location answer
        # Select object A: the position in the image, if there is only one object A in the image, the position can be omitted, the position is to distinguish different objects, the position relative to the image
        self.dis_loc_ans_loc_a.addItems(
            [
                "Position A",
                "none",
                "top",
                "bottom",
                "left",
                "right",
                "center",
                "top left",
                "top right",
                "bottom left",
                "bottom right",
            ]
        )
        self.dis_loc_ans_loc_a.setGeometry(X_base + 37, 180, 120 + 30, 23)
        self.dis_loc_ans_loc_a.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        self.dis_loc_is_cluster_a.addItems(["Is it a cluster?", "none", "cluster"])
        self.dis_loc_is_cluster_a.setGeometry(X_base + 162 + 30, 180, 80, 23)
        self.dis_loc_is_cluster_a.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # self.dis_loc_ans_loc_a.currentIndexChanged.connect(self.get_object_one)

        self.dis_loc_ans_object_a.addItems(["Object A"] + self.subset_list)
        self.dis_loc_ans_object_a.setGeometry(X_base + 247 + 30, 180, 150 + 80, 23)
        self.dis_loc_ans_object_a.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # self.dis_loc_ans_object_a.currentIndexChanged.connect(self.get_object_one)

        # Select object B: the position in the image
        self.dis_loc_ans_loc_b.addItems(
            [
                "Position B",
                "none",
                "top",
                "bottom",
                "left",
                "right",
                "center",
                "top left",
                "top right",
                "bottom left",
                "bottom right",
            ]
        )
        self.dis_loc_ans_loc_b.setGeometry(X_base + 37, 213, 120 + 30, 23)
        self.dis_loc_ans_loc_b.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # self.dis_loc_ans_loc_b.currentIndexChanged.connect(self.get_object_one)
        self.dis_loc_is_cluster_b.addItems(["Is it a cluster?", "none", "cluster"])
        self.dis_loc_is_cluster_b.setGeometry(X_base + 162 + 30, 213, 80, 23)
        self.dis_loc_is_cluster_b.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        self.dis_loc_ans_object_b.addItems(["Object B"] + self.subset_list)
        self.dis_loc_ans_object_b.setGeometry(X_base + 247 + 30, 213, 150 + 80, 23)
        self.dis_loc_ans_object_b.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # self.dis_loc_ans_object_b.currentIndexChanged.connect(self.get_object_two)

        # The distance between object AB, take the nearest linear distance between the two objects, next to means adjacent  
        self.dis_loc_ans_distance.addItems(
            [
                "A-B Distance",
                "none",
                "next to",
                "0-25",
                "25-50",
                "50-75",
                "75-100",
                "100-125",
                "125-150",
                "150-175",
                "175-200",
                "200-225",
                "225-250",
                "250-275",
                "275-300",
                "300-325",
                "325-350",
                "350-375",
                "375-400",
            ]
        )
        self.dis_loc_ans_distance.setGeometry(X_base + 400 + 110, 180, 100 + 20, 23)
        self.dis_loc_ans_distance.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # self.dis_loc_ans_distance.currentIndexChanged.connect(self.get_object_two)
        # The position of B relative to A
        self.dis_loc_ans_location.addItems(
            [
                "B's position relative to A",
                "above",
                "below",
                "left",
                "right",
                "upper left",
                "upper right",
                "bottom left",
                "bottom right",
            ]
        )
        self.dis_loc_ans_location.setGeometry(X_base + 400 + 110, 213, 100 + 20, 23)
        self.dis_loc_ans_location.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # self.dis_loc_ans_location.currentIndexChanged.connect(self.get_object_two)

        # Submit position annotation button
        self.submit_dis_loc = QtWidgets.QPushButton("Confirm", self)
        self.submit_dis_loc.setGeometry(X_base + 167, 246, 120, 23)
        self.submit_dis_loc.pressed.connect(self.join_dis_loc_dict)
        # Distance measurement button
        self.open_dis_loc = QtWidgets.QPushButton("Length and Distance", self)
        self.open_dis_loc.setGeometry(X_base + 37, 246, 120, 23)
        self.open_dis_loc.pressed.connect(self.get_distance)
        # Generate questions to buffer
        self.generate_dis_loc_btn = QtWidgets.QPushButton("Finish Input", self)
        self.generate_dis_loc_btn.setGeometry(X_base + 297, 246, 120, 23)
        self.generate_dis_loc_btn.pressed.connect(self.generate_dis_loc)

        self.contain_ans_landcover.addItems(["LandCover Class"] + self.landcover_list)
        self.contain_ans_landcover.setGeometry(X_base + 37, 330, 120 + 80, 23)
        self.contain_ans_landcover.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.contain_ans_landcover.currentIndexChanged.connect(self.get_subset)
        self.contain_ans_landcover.currentIndexChanged.connect(
            self.get_current_landcover
        )

        self.contain_ans_subset.addItems(["Subset Class"])
        self.contain_ans_subset.setGeometry(X_base + 167 + 80, 330, 120 + 80, 23)
        self.contain_ans_subset.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.contain_ans_subset.currentIndexChanged.connect(
            self.show_pre_contain_attribution
        )

        # self.contain_ans_number.addItems(['Number','none','1','2','3','4','5','5-10','10-20','20-40','40-100','> 100'])
        self.contain_ans_number.addItems(
            [
                "Number",
                "none",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "6-10",
                "10-20",
                "20-40",
                "40-100",
                "> 100",
            ]
        )  # 2024/1/13
        self.contain_ans_number.setGeometry(X_base + 297 + 160, 330, 120 + 20, 23)
        self.contain_ans_number.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # The overall position in the image
        self.contain_ans_location.addItems(
            [
                "Location",
                "none",
                "top",
                "bottom",
                "left",
                "right",
                "central",
                "upper left",
                "upper right",
                "lower left",
                "lower right",
                "almost all the picture",
            ]
        )
        self.contain_ans_location.setGeometry(X_base + 37, 363, 120 + 80, 23)
        self.contain_ans_location.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # The overall shape
        self.contain_ans_shape.addItems(
            [
                "Shape",
                "none",
                "Straight",
                "Curved",
                "Triangle",
                "Square",
                "Rectangle",
                "other quadrilater",
                "Rotundity",
                "other shape",
            ]
        )
        self.contain_ans_shape.setGeometry(X_base + 167 + 80, 363, 120 + 80, 23)
        self.contain_ans_shape.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        # Area
        self.contain_ans_Area.addItems(
            [
                "Area",
                "none",
                "0-100",
                "100-200",
                "200-300",
                "300-400",
                "400-500",
                "500-600",
                "600-700",
                "700-800",
                "800-900",
                "900-1000",
                "1000-2000",
                "2000-3000",
                "3000-4000",
                "4000-5000",
                "5000-6000",
                "6000-7000",
                "7000-8000",
                "8000-9000",
                "9000-10000",
                "10000-12500",
                "12500-15000",
                "15000-17500",
                "17500-20000",
                "20000-22500",
                "22500-25000",
                "25000-27500",
                "27500-30000",
                "30000-32500",
                "32500-35000",
                "35000-37500",
                "37500-40000",
                "40000-50000",
                "50000-60000",
                "more than 60000",
            ]
        )
        self.contain_ans_Area.setGeometry(X_base + 297 + 160, 363, 120 + 20, 23)
        self.contain_ans_Area.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Length
        self.contain_ans_length.addItems(
            [
                "Length",
                "none",
                "0-25",
                "25-50",
                "50-75",
                "75-100",
                "100-125",
                "125-150",
                "150-175",
                "175-200",
                "200-225",
                "225-250",
                "250-275",
                "275-300",
                "300-325",
                "325-350",
                "350-375",
                "375-400",
            ]
        )
        self.contain_ans_length.setGeometry(X_base + 37, 396, 120 + 80, 23)
        self.contain_ans_length.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Distribution, aggregation or dispersion
        self.contain_ans_distribution.addItems(
            [
                "Distribution",
                "none",
                "Clustered Distribution",
                "Isolated Distribution",
                "Dense Distribution",
                "Random distribution",
                "Uniform distribution",
            ]
        )
        self.contain_ans_distribution.setGeometry(X_base + 167 + 80, 396, 120 + 80, 23)
        self.contain_ans_distribution.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Modal better
        self.contain_ans_better.addItems(
            ["Better", "optical", "thermal", "almost same"]
        )
        self.contain_ans_better.setGeometry(X_base + 297 + 160, 396, 120 + 20, 23)
        self.contain_ans_better.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Confirm, calculate area, generate questions
        self.submit_contain.setGeometry(X_base + 167, 429, 120, 23)
        self.submit_contain.pressed.connect(self.join_contain)

        self.open_contain = QtWidgets.QPushButton("Calculate Area", self)
        self.open_contain.setGeometry(X_base + 37, 429, 120, 23)
        self.open_contain.pressed.connect(self.get_area)

        self.generate_contain_btn = QtWidgets.QPushButton("Finish Input", self)
        self.generate_contain_btn.setGeometry(X_base + 297, 429, 120, 23)
        self.generate_contain_btn.pressed.connect(self.generate_contain)
        # Delete current landcover
        self.del_contain_btn = QtWidgets.QPushButton("Delete Current LandCover", self)
        self.del_contain_btn.setGeometry(X_base + 427, 429, 150, 23)
        self.del_contain_btn.pressed.connect(self.del_pres_contain_attribution)

        # Traffic question answer
        # Traffic LandCover
        self.traffic_ans_landcover.addItems(
            ["LandCover Class"] + self.traffic_landcover_list
        )
        self.traffic_ans_landcover.setGeometry(X_base + 37, 340, 120 + 80, 23)
        self.traffic_ans_landcover.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.traffic_ans_landcover.currentIndexChanged.connect(self.get_traffic_object)
        self.traffic_ans_landcover.currentIndexChanged.connect(
            self.get_current_traffic_landcover
        )

        # Traffic object
        self.traffic_ans_object.addItems(["Subset Class"])
        self.traffic_ans_object.setGeometry(X_base + 167 + 80, 340, 120 + 80, 23)
        self.traffic_ans_object.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.traffic_ans_object.currentIndexChanged.connect(
            self.show_pre_traffic_attribution
        )

        # Traffic number
        self.traffic_ans_number.addItems(
            [
                "Number",
                "none",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "6-10",
                "10-20",
                "20-40",
                "40-100",
                "> 100",
            ]
        )
        self.traffic_ans_number.setGeometry(X_base + 297 + 160, 340, 120 + 20, 23)
        self.traffic_ans_number.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Traffic location
        self.traffic_ans_location.addItems(
            [
                "Location",
                "none",
                "top",
                "bottom",
                "left",
                "right",
                "central",
                "upper left",
                "upper right",
                "lower left",
                "lower right",
                "almost all the picture",
            ]
        )
        self.traffic_ans_location.setGeometry(X_base + 37, 373, 120 + 80, 23)
        self.traffic_ans_location.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Traffic better
        self.traffic_ans_better.addItems(
            ["Quality", "optical", "thermal", "almost same"]
        )
        self.traffic_ans_better.setGeometry(X_base + 167 + 80, 373, 120 + 80, 23)
        self.traffic_ans_better.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Confirm input
        self.submit_traffic.setGeometry(X_base + 167, 373, 120, 23)
        self.submit_traffic.pressed.connect(self.join_traffic)

        # Finish input
        self.generate_traffic_btn = QtWidgets.QPushButton("Finish Input", self)
        self.generate_traffic_btn.setGeometry(X_base + 297, 373, 120, 23)
        self.generate_traffic_btn.pressed.connect(self.generate_traffic)

        # Delete current traffic landcover
        self.del_traffic_btn = QtWidgets.QPushButton("Delete Current Traffic LandCover", self)
        self.del_traffic_btn.setGeometry(X_base + 427, 373, 150, 23)
        self.del_traffic_btn.pressed.connect(self.del_traffic_attribution)

        # Residential question answer
        # Residential Landcover
        self.residential_ans_landcover.addItems(
            ["LandCover Class"] + self.residential_landcover_list
        )
        self.residential_ans_landcover.setGeometry(X_base + 37, 380, 120 + 80, 23)
        self.residential_ans_landcover.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.residential_ans_landcover.currentIndexChanged.connect(
            self.get_residential_object
        )
        self.residential_ans_landcover.currentIndexChanged.connect(
            self.get_current_residential_landcover
        )

        # Residential object
        self.residential_ans_object.addItems(
            ["Subset Class"] + self.residential_object_list
        )
        self.residential_ans_object.setGeometry(X_base + 167 + 80, 380, 120 + 80, 23)
        self.residential_ans_object.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )
        self.residential_ans_object.currentIndexChanged.connect(
            self.show_pre_residential_attribution
        )

        # Residential number
        self.residential_ans_number.addItems(
            [
                "Number",
                "none",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "6-10",
                "10-20",
                "20-40",
                "40-100",
                "> 100",
            ]
        )
        self.residential_ans_number.setGeometry(X_base + 297 + 160, 380, 120 + 20, 23)
        self.residential_ans_number.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Residential location
        # self.residential_ans_location.addItems(
        #     [
        #         "Location",
        #         "top interior",
        #         "bottom interior",
        #         "left interior",
        #         "right interior",
        #         "top-left interior",
        #         "top-right interior",
        #         "bottom-left interior",
        #         "bottom-right interior",
        #         "central interior",
        #         "top exterior",
        #         "bottom exterior",
        #         "left exterior",
        #         "right exterior",
        #         "top-left exterior",
        #         "top-right exterior",
        #         "bottom-left exterior",
        #         "bottom-right exterior",
        #         "almost all the picture",
        #     ]
        # )
        self.residential_ans_location.addItems(
            [
                "Location",
                "none",
                "top",
                "bottom",
                "left",
                "right",
                "central",
                "upper left",
                "upper right",
                "lower left",
                "lower right",
                "almost all the picture",
            ]
        )
        self.residential_ans_location.setGeometry(X_base + 37, 413, 120 + 80, 23)
        self.residential_ans_location.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Residential better
        self.residential_ans_better.addItems(
            ["Quality", "optical", "thermal", "almost same"]
        )
        self.residential_ans_better.setGeometry(X_base + 167 + 80, 413, 120 + 80, 23)
        self.residential_ans_better.setStyleSheet(
            "font-size: 16px;font-weight: bold;background-color: white"
        )

        # Confirm input
        self.submit_residential.setGeometry(X_base + 167, 413, 120, 23)
        self.submit_residential.pressed.connect(self.join_residential)

        # Finish input
        self.generate_residential_btn = QtWidgets.QPushButton("Finish Input", self)
        self.generate_residential_btn.setGeometry(X_base + 297, 413, 120, 23)
        self.generate_residential_btn.pressed.connect(self.generate_residential)

        # Delete current residential landcover
        self.del_residential_btn = QtWidgets.QPushButton(
            "Delete Current Residential LandCover", self
        )
        self.del_residential_btn.setGeometry(X_base + 427, 413, 150, 23)
        self.del_residential_btn.pressed.connect(self.del_residential_attribution)

        # Deduce question: give your own question and answer
        self.Deduce_Qst_one.setGeometry(X_base + 37, 500, 320, 23)
        self.Deduce_Qst_one.setStyleSheet("font-size:15px")
        self.Deduce_Ans_one.setGeometry(X_base + 365, 500, 80, 23)
        self.Deduce_Ans_one.setStyleSheet("font-size:15px")

        self.Deduce_Qst_two.setGeometry(X_base + 37, 530, 320, 23)
        self.Deduce_Qst_two.setStyleSheet("font-size:15px")
        self.Deduce_Ans_two.setGeometry(X_base + 365, 530, 80, 23)
        self.Deduce_Ans_two.setStyleSheet("font-size:15px")

        # Question submit button
        self.submit_one_btn = QtWidgets.QPushButton("submit first", self)
        self.submit_one_btn.setGeometry(X_base + 37, 565, 140, 25)
        self.submit_one_btn.setStyleSheet("font-size:17px;font-weight:bold")
        self.submit_one_btn.pressed.connect(self.ans_deduce_one)

        self.submit_two_btn = QtWidgets.QPushButton("submit second", self)
        self.submit_two_btn.setGeometry(X_base + 217, 565, 140, 25)
        self.submit_two_btn.setStyleSheet("font-size:17px;font-weight:bold")
        self.submit_two_btn.pressed.connect(self.ans_deduce_two)

        # Switch image button

        # Previous image: Pre
        self.Pre_btn.setGeometry(X_base + 37, 650, 80, 30)
        self.Pre_btn.setStyleSheet("font-size:19px;font-weight:bold")
        self.Pre_btn.clicked.connect(self.show_prev_image)

        # Next image: Next
        self.Next_btn.setGeometry(X_base + 137, 650, 80, 30)
        self.Next_btn.setStyleSheet("font-size:19px;font-weight:bold")
        self.Next_btn.clicked.connect(self.show_next_image)

        # Submit current file
        self.Sub_btn.setGeometry(X_base + 237, 650, 80, 30)
        self.Sub_btn.setStyleSheet("font-size:19px;font-weight:bold")
        self.Sub_btn.clicked.connect(self.submit_annotation)
        # Generate annotation file
        self.Gen_btn.setGeometry(X_base + 337, 650, 100, 30)
        self.Gen_btn.setStyleSheet("font-size:19px;font-weight:bold")
        self.Gen_btn.clicked.connect(self.generate_annotation)

        # Add question and answer widgets to the scroll area
        self.scroll_layout.addWidget(self.question_headline)
        self.scroll_layout.addWidget(self.question_Match)
        self.scroll_layout.addWidget(self.Match_pick_box)
        self.scroll_layout.addWidget(self.Mist_pick_box)
        self.scroll_layout.addWidget(self.night_pick_box)

        self.scroll_layout.addWidget(self.question_theme)
        self.scroll_layout.addWidget(self.theme_ans_r)
        self.scroll_layout.addWidget(self.theme_ans_ur)

        self.scroll_layout.addWidget(self.question_dis_loc)
        self.scroll_layout.addWidget(self.dis_loc_ans_loc_a)
        self.scroll_layout.addWidget(self.dis_loc_is_cluster_a)
        self.scroll_layout.addWidget(self.dis_loc_ans_object_a)
        self.scroll_layout.addWidget(self.dis_loc_ans_loc_b)
        self.scroll_layout.addWidget(self.dis_loc_is_cluster_b)
        self.scroll_layout.addWidget(self.dis_loc_ans_object_b)
        self.scroll_layout.addWidget(self.dis_loc_ans_distance)
        self.scroll_layout.addWidget(self.dis_loc_ans_location)
        self.scroll_layout.addWidget(self.submit_dis_loc)
        self.scroll_layout.addWidget(self.open_dis_loc)
        self.scroll_layout.addWidget(self.generate_dis_loc_btn)

        self.scroll_layout.addWidget(self.question_contain)
        self.scroll_layout.addWidget(self.contain_ans_landcover)
        self.scroll_layout.addWidget(self.contain_ans_subset)
        self.scroll_layout.addWidget(self.contain_ans_number)
        self.scroll_layout.addWidget(self.contain_ans_location)
        self.scroll_layout.addWidget(self.contain_ans_shape)
        self.scroll_layout.addWidget(self.contain_ans_Area)
        self.scroll_layout.addWidget(self.contain_ans_length)
        self.scroll_layout.addWidget(self.contain_ans_distribution)
        self.scroll_layout.addWidget(self.contain_ans_better)
        self.scroll_layout.addWidget(self.submit_contain)
        self.scroll_layout.addWidget(self.open_contain)
        self.scroll_layout.addWidget(self.generate_contain_btn)
        self.scroll_layout.addWidget(self.del_contain_btn)

        self.scroll_layout.addWidget(self.question_deduce)
        self.scroll_layout.addWidget(self.Deduce_Qst_one)
        self.scroll_layout.addWidget(self.Deduce_Ans_one)
        self.scroll_layout.addWidget(self.Deduce_Qst_two)
        self.scroll_layout.addWidget(self.Deduce_Ans_two)
        self.scroll_layout.addWidget(self.submit_one_btn)
        self.scroll_layout.addWidget(self.submit_two_btn)

        self.scroll_layout.addWidget(self.question_traffic)
        self.scroll_layout.addWidget(self.traffic_ans_landcover)
        self.scroll_layout.addWidget(self.traffic_ans_object)
        self.scroll_layout.addWidget(self.traffic_ans_number)
        self.scroll_layout.addWidget(self.traffic_ans_location)
        self.scroll_layout.addWidget(self.traffic_ans_better)
        self.scroll_layout.addWidget(self.submit_traffic)
        self.scroll_layout.addWidget(self.generate_traffic_btn)
        self.scroll_layout.addWidget(self.del_traffic_btn)

        self.scroll_layout.addWidget(self.question_residential)
        self.scroll_layout.addWidget(self.residential_ans_landcover)
        self.scroll_layout.addWidget(self.residential_ans_object)
        self.scroll_layout.addWidget(self.residential_ans_number)
        self.scroll_layout.addWidget(self.residential_ans_location)
        self.scroll_layout.addWidget(self.residential_ans_better)
        self.scroll_layout.addWidget(self.submit_residential)
        self.scroll_layout.addWidget(self.generate_residential_btn)
        self.scroll_layout.addWidget(self.del_residential_btn)

        self.scroll_layout.addWidget(self.question_agricultural)
        self.scroll_layout.addWidget(self.agricultural_ans_road)
        self.scroll_layout.addWidget(self.agricultural_ans_water)

        self.scroll_layout.addWidget(self.question_industrial)
        self.scroll_layout.addWidget(self.industrial_ans_facility)
        self.scroll_layout.addWidget(self.industrial_ans_scale)
        self.scroll_layout.addWidget(self.industrial_ans_location)

        self.scroll_layout.addWidget(self.question_uav)
        self.scroll_layout.addWidget(self.uav_ans_height)
        self.scroll_layout.addWidget(self.uav_ans_angle)

        self.scroll_layout.addWidget(self.Pre_btn)
        self.scroll_layout.addWidget(self.Next_btn)
        self.scroll_layout.addWidget(self.Sub_btn)
        self.scroll_layout.addWidget(self.Gen_btn)

        # Style settings
        self.setStyleSheet(
            "QPushButton#blueButton:hover:pressed {background-color: #0D47A1;}"
            "QLabel#selectedFolderLabel {background-color: white;border: 1px solid #aaa;font-size:15px}"
            "QLabel#headline {font-weight: bold;font-size: 15px}"
            "QCheckbox#Qst {font-size:19px;font-family: SimHei}"
        )

    # Join attribute dictionary and update attribute dictionary
    def join_attribute_dict(self, key, value):
        if self.flag == True:
            if len(value) > 0:
                self.current_attri_dict[key] = value
                self.display_attribution()

    def del_attribute_dict(self, key):
        if self.flag == True:
            if key in self.current_attri_dict:
                self.current_attri_dict.pop(key)
                self.display_attribution()

    def del_pres_contain_attribution(self):
        if self.flag == True:
            if "PresContain" in self.current_attri_dict:
                if self.current_landcover in self.current_attri_dict["PresContain"]:
                    self.current_attri_dict["PresContain"].pop(self.current_landcover)
        # Display the attribute information of the submitted objects
        if len(self.contain_dict) > 0:
            self.display_anno.clear()
            for k, v in self.contain_dict.items():
                if len(v) > 0:
                    for s in v:
                        self.display_anno.append(
                            "Con&&Pre (%s:%s): Num %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s"
                            % (
                                k,
                                s[0][1],
                                s[1][1],
                                s[2][1],
                                s[3][1],
                                s[4][1],
                                s[5][1],
                                s[6][1],
                                s[7][1],
                            )
                        )

    def del_traffic_attribution(self):
        if self.flag == True:
            if "Traffic" in self.current_attri_dict:
                if self.current_traffic_landcover in self.current_attri_dict["Traffic"]:
                    self.current_attri_dict["Traffic"].pop(
                        self.current_traffic_landcover
                    )
        # Update display
        if len(self.traffic_dict) > 0:
            self.display_anno.clear()
            for k, v in self.traffic_dict.items():
                if len(v) > 0:
                    for s in v:
                        # Display traffic annotation, Num, Loc, Better
                        self.display_anno.append(
                            "Traffic (%s:%s): Num %s, Loc %s, Better %s"
                            % (k, s[0][1], s[1][1], s[2][1], s[3][1])
                        )

    def del_residential_attribution(self):
        if self.flag == True:
            if "Residential" in self.current_attri_dict:
                if (
                    self.current_residential_landcover
                    in self.current_attri_dict["Residential"]
                ):
                    self.current_attri_dict["Residential"].pop(
                        self.current_residential_landcover
                    )
        if len(self.residential_dict) > 0:
            self.display_anno.clear()
            for k, v in self.residential_dict.items():
                if len(v) > 0:
                    for s in v:
                        self.display_anno.append(
                            "Residential (%s:%s): Num %s, Loc %s, Better %s"
                            % (k, s[0][1], s[1][1], s[2][1], s[3][1])
                        )

    # Define the function corresponding to each question
    def ans_match(self):
        # The checkbox of match is selected, and the answer in self.Match_pick_box is selected
        # self.Match_pick_box.addItems(['Please choose','Yes','No','Almost match'])
        if self.question_Match.isChecked() and self.Match_pick_box.currentIndex() != 0:

            tri_list = []
            qst = "How is the matching situation in these two pictures ?"
            ans = self.Match_pick_box.currentText()
            tri_list.append(qst)
            tri_list.append(ans)
            self.jion_current_dict("11", tri_list)
            self.join_attribute_dict("match", ans)

        else:
            self.del_annotation(q_id="11")
            self.del_attribute_dict("match")

        if self.question_Match.isChecked() and self.Mist_pick_box.currentIndex() != 0:
            ans = self.Mist_pick_box.currentText()
            self.join_attribute_dict("mist", ans)
        else:
            self.del_attribute_dict("mist")

        if self.question_Match.isChecked() and self.night_pick_box.currentIndex() != 0:
            ans = self.night_pick_box.currentText()
            self.join_attribute_dict("night", ans)
        else:
            self.del_attribute_dict("night")

        

    def ans_theme(self):
        if self.question_theme.isChecked() and self.theme_ans_r.currentIndex() != 0:
            tri_list_1 = []  # Store questions and answers
            tri_list_2 = []
            qst_1 = "What is the theme of this picture ?"
            qst_2 = "Is it a residential area or a non-residential area ?"
            if self.theme_ans_r.currentIndex() == 1:
                ans = "residential area"
            else:
                ans = "non-residential area"
            tri_list_1.append(qst_1)
            tri_list_1.append(ans)

            tri_list_2.append(qst_2)
            tri_list_2.append(ans)
            self.jion_current_dict("21", tri_list_1)
            self.jion_current_dict("22", tri_list_2)
            self.join_attribute_dict("theme", ans)

        else:
            self.del_annotation(q_id="21")
            self.del_annotation(q_id="22")
            self.del_attribute_dict("theme")

        if self.question_theme.isChecked() and self.theme_ans_ur.currentIndex() != 0:
            tri_list = []
            qst = "Is it a rural or an urban area ?"
            ans = self.theme_ans_ur.currentText()
            tri_list.append(qst)
            tri_list.append(ans)
            self.jion_current_dict("23", tri_list)
            self.join_attribute_dict("urban", ans)
        else:
            self.del_annotation(q_id="23")
            self.del_attribute_dict("urban")

    def del_dis_loc(self):
        if not self.question_dis_loc.isChecked():
            self.del_annotation(q="3")
            self.del_attribute_dict("LocDis")
            self.dis_loc_dict = {}

    def ans_deduce_one(self):
        if (
            self.question_deduce.isChecked()
            and self.Deduce_Qst_one.text() not in "Enter your question 1 (one-word answer)"
            and self.Deduce_Ans_one.text() not in "Enter answer"
        ):
            qst = self.Deduce_Qst_one.text()
            ans = self.Deduce_Ans_one.text()
            tri_list = [qst, ans]
            self.join_attribute_dict("Deduce_one", tri_list)
            self.jion_current_dict("51", tri_list)

    def ans_deduce_two(self):
        if (
            self.question_deduce.isChecked()
            and self.Deduce_Qst_two.text() not in "Enter your question 2 (one-word answer)"
            and self.Deduce_Ans_two.text() not in "Enter answer"
        ):
            qst = self.Deduce_Qst_two.text()
            ans = self.Deduce_Ans_two.text()
            tri_list = [qst, ans]
            self.join_attribute_dict("Deduce_two", tri_list)
            self.jion_current_dict("52", tri_list)

    def del_deduce(self):
        if not self.question_deduce.isChecked():
            self.del_annotation(q="5")
            self.del_attribute_dict("Deduce_one")
            self.del_attribute_dict("Deduce_two")

    # Agricultural area question
    def ans_agricultural(self):
        if self.question_agricultural.isChecked():
            if self.agricultural_ans_road.currentIndex() != 0:
                tri_list = []
                qst = "Is there a road in the picture ?"
                ans = self.agricultural_ans_road.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("61", tri_list)
                self.join_attribute_dict("agricultural_road", ans)
            else:
                self.del_annotation(q_id="61")
                self.del_attribute_dict("agricultural_road")

            if self.agricultural_ans_water.currentIndex() != 0:
                tri_list = []
                qst = "Is there water in the picture ?"
                ans = self.agricultural_ans_water.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("62", tri_list)
                self.join_attribute_dict("water_source", ans)
            else:
                self.del_annotation(q_id="62")
                self.del_attribute_dict("water_source")
        else:
            self.del_annotation(q_id="61")
            self.del_annotation(q_id="62")
            self.del_attribute_dict("agricultural_road")
            self.del_attribute_dict("water_source")

    # Industrial area question
    def ans_industrial(self):
        if self.question_industrial.isChecked():
            if self.industrial_ans_facility.currentIndex() != 0:
                tri_list = []
                qst = "Is there a facility in the picture ?"
                ans = self.industrial_ans_facility.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("71", tri_list)
                self.join_attribute_dict("logistics_facility", ans)
            else:
                self.del_annotation(q_id="71")
                self.del_attribute_dict("logistics_facility")

            if self.industrial_ans_scale.currentIndex() != 0:
                tri_list = []
                qst = "What is the scale of the facility ?"
                ans = self.industrial_ans_scale.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("72", tri_list)
                self.join_attribute_dict("construction_scale", ans)
            else:
                self.del_annotation(q_id="72")
                self.del_attribute_dict("construction_scale")

            if self.industrial_ans_location.currentIndex() != 0:
                tri_list = []
                qst = "What is the location of the industrial area ?"
                ans = self.industrial_ans_location.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("73", tri_list)
                self.join_attribute_dict("industrial_location", ans)
            else:
                self.del_annotation(q_id="73")
                self.del_attribute_dict("industrial_location")
        else:
            self.del_annotation(q_id="71")
            self.del_annotation(q_id="72")
            self.del_annotation(q_id="73")
            self.del_attribute_dict("logistics_facility")
            self.del_attribute_dict("construction_scale")
            self.del_attribute_dict("industrial_location")

    # UAV question
    def ans_uav(self):
        if self.question_uav.isChecked():
            if self.uav_ans_height.currentIndex() != 0:
                tri_list = []
                qst = "What is the height of the UAV ?"
                ans = self.uav_ans_height.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("81", tri_list)
                self.join_attribute_dict("UAV_height", ans)
            else:
                self.del_annotation(q_id="81")
                self.del_attribute_dict("UAV_height")

            if self.uav_ans_angle.currentIndex() != 0:
                tri_list = []
                qst = "What is the angle of the UAV ?"
                ans = self.uav_ans_angle.currentText()
                tri_list.append(qst)
                tri_list.append(ans)
                self.jion_current_dict("82", tri_list)
                self.join_attribute_dict("UAV_angle", ans)
            else:
                self.del_annotation(q_id="82")
                self.del_attribute_dict("UAV_angle")
        else:
            self.del_annotation(q_id="81")
            self.del_annotation(q_id="82")
            self.del_attribute_dict("UAV_height")
            self.del_attribute_dict("UAV_angle")

    def all_Index_Check(self, n=1):
        if n == 1:
            return (
                self.dis_loc_ans_loc_a.currentIndex() != 0
                and self.dis_loc_ans_object_a.currentIndex() != 0
                and self.dis_loc_ans_loc_b.currentIndex() != 0
                and self.dis_loc_ans_object_b.currentIndex() != 0
                and self.dis_loc_ans_distance.currentIndex() != 0
                and self.dis_loc_ans_location.currentIndex() != 0
                and self.dis_loc_is_cluster_b.currentIndex() != 0
                and self.dis_loc_is_cluster_a.currentIndex() != 0
            )
        elif n == 2:
            return (
                self.contain_list[0].currentIndex() != 0
                and self.contain_list[1].currentIndex() != 0
                and self.contain_list[2].currentIndex() != 0
                and self.contain_list[3].currentIndex() != 0
                and self.contain_list[4].currentIndex() != 0
                and self.contain_list[5].currentIndex() != 0
                and self.contain_list[6].currentIndex() != 0
                and self.contain_list[7].currentIndex() != 0
                and self.contain_list[8].currentIndex() != 0
            )
        elif n == 3:
            return (
                self.traffic_list[0].currentIndex() != 0
                and self.traffic_list[1].currentIndex() != 0
                and self.traffic_list[2].currentIndex() != 0
                and self.traffic_list[3].currentIndex() != 0
                and self.traffic_list[4].currentIndex() != 0
            )
        elif n == 4:
            return (
                self.residential_list[0].currentIndex() != 0
                and self.residential_list[1].currentIndex() != 0
                and self.residential_list[2].currentIndex() != 0
                and self.residential_list[3].currentIndex() != 0
                and self.residential_list[4].currentIndex() != 0
            )

    # Record the position information of A and B in the image, if there is only one object, select location as none
    def join_dis_loc_dict(self):
        if self.question_dis_loc.isChecked() and self.all_Index_Check():
            if self.dis_loc_ans_loc_a.currentIndex() == 1:
                if self.dis_loc_is_cluster_a.currentIndex() == 1:
                    A = self.dis_loc_ans_object_a.currentText()
                else:
                    A = "a cluster of " + self.dis_loc_ans_object_a.currentText()
            else:
                if self.dis_loc_is_cluster_a.currentIndex() == 1:
                    A = (
                        self.dis_loc_ans_object_a.currentText()
                        + "s at the %s of the picture"
                        % self.dis_loc_ans_loc_a.currentText()
                    )
                else:
                    A = (
                        "a cluster of "
                        + self.dis_loc_ans_object_a.currentText()
                        + "s at the %s of the picture"
                        % self.dis_loc_ans_loc_a.currentText()
                    )

            if self.dis_loc_ans_loc_b.currentIndex() == 1:
                if self.dis_loc_is_cluster_b.currentIndex() == 1:
                    B = self.dis_loc_ans_object_b.currentText()
                else:
                    B = "a cluster of " + self.dis_loc_ans_object_b.currentText()
            else:
                # A and B have already added location information.
                if self.dis_loc_is_cluster_b.currentIndex() == 1:
                    # self.dis_loc_dict:{(A,B):[dis_index,location,distance],(C,D):[dis_index,location,distance],...}
                    B = (
                        self.dis_loc_ans_object_b.currentText()
                        + "s at the %s of the picture"
                        % self.dis_loc_ans_loc_b.currentText()
                    )
                else:
                    B = (
                        "a cluster of "
                        + self.dis_loc_ans_object_b.currentText()
                        + "s at the %s of the picture"
                        % self.dis_loc_ans_loc_b.currentText()
                    )

            self.dis_loc_dict[(A, B)] = [
                self.dis_loc_ans_distance.currentIndex(),
                self.dis_loc_ans_location.currentText(),
                self.dis_loc_ans_distance.currentText(),
            ]
            self.dis_loc_ans_distance.setCurrentIndex(0)
            self.dis_loc_ans_location.setCurrentIndex(0)

            # Display the distance and relative position of the submitted distance, helping reference

            if len(self.dis_loc_dict) > 0:
                # self.display_anno.clear()
                self.display_anno.clear()
                for k, v in self.dis_loc_dict.items():
                    self.display_anno.append(
                        "Dis&&Loc (%s , %s): Loc %s, Dis %s" % (k[0], k[1], v[1], v[2])
                    )
        else:
            self.Warn_Erorr("Some options in Distance/Location are not selected, cannot confirm")

    # Location dictionary

    def generate_dis_loc(self):
        location2location = {
            "above": "above",
            "below": "below",
            "left": "to the left of",
            "right": "to the right of",
            "upper left": "in the upper left of",
            "upper right": "in the upper right of",
            "bottom left": "at the bottom left of",
            "bottom right": "at the bottom right of",
        }
        ab_location2location = {
            "above": "below",
            "below": "above",
            "left": "to the right of",
            "right": "to the left of",
            "upper left": "in the bottom right of",
            "upper right": "in the bottom left of",
            "bottom left": "at the upper right of",
            "bottom right": "at the upper left of",
        }
        if self.question_dis_loc.isChecked() and len(self.dis_loc_dict) > 0:
            # Store all object pairs with distance and location relationships [[A,B,dis_index,location,distance],[C,D,dis_index,location,distance],...]
            dis_loc_list = []
            counter = 1
            for k, v in self.dis_loc_dict.items():
                dis_loc_list.append([k[0], k[1], v[0], v[1], v[2]])
            for i in range(len(dis_loc_list)):
                object_A = dis_loc_list[i][0]
                object_B = dis_loc_list[i][1]
                index = dis_loc_list[i][2]
                location = dis_loc_list[i][3]
                distance = dis_loc_list[i][4]

                qst_one = "What is the distance between the %s and the %s ?" % (
                    object_A,
                    object_B,
                )
                ans_one = distance

                qst_two = "What is the position of the %s relative to the %s ?" % (
                    object_B,
                    object_A,
                )
                ans_two = location

                qst_three = "Is the %s %s the %s ?" % (
                    object_B,
                    location2location[location],
                    object_A,
                )
                ans_three = "Yes"

                qst_four = "Is the %s %s the %s ?" % (
                    object_A,
                    location2location[location],
                    object_B,
                )
                ans_four = "No"

                qst_five = "What is the position of the %s relative to the %s ?" % (
                    object_A,
                    object_B,
                )
                ans_five = self.ablocation[location]

                qst_six = "Is the %s %s the %s ?" % (
                    object_A,
                    ab_location2location[location],
                    object_B,
                )
                ans_six = "Yes"

                qst_seven = "Is the %s %s the %s ?" % (
                    object_B,
                    ab_location2location[location],
                    object_A,
                )
                ans_seven = "No"

                qst_ans_list = [
                    [qst_one, ans_one],
                    [qst_two, ans_two],
                    [qst_three, ans_three],
                    [qst_four, ans_four],
                    [qst_five, ans_five],
                    [qst_six, ans_six],
                    [qst_seven, ans_seven],
                ]
                for qst in qst_ans_list:
                    self.jion_current_dict("3" + str(counter), qst)
                    counter = counter + 1

                while i + 1 < len(dis_loc_list):
                    object_A_ = dis_loc_list[i + 1][0]
                    object_B_ = dis_loc_list[i + 1][1]
                    index_ = dis_loc_list[i + 1][2]
                    qst_ans_list_ = []
                    qst_eight = (
                        "Is the distance between the %s and the %s longer than the distance between the %s and the %s ?"
                        % (object_A, object_B, object_A_, object_B_)
                    )
                    qst_nine = (
                        "Is the distance between the %s and the %s longer than the distance between the %s and the %s ?"
                        % (object_A_, object_B_, object_A, object_B)
                    )
                    qst_ten = (
                        "How is the distance between the %s and the %s compared to the distance between the %s and the %s ?"
                        % (object_A, object_B, object_A_, object_B_)
                    )
                    if index > index_:
                        ans_eight = "Yes"
                        ans_nine = "No"
                        ans_ten = "longer"
                    elif index == index_:
                        ans_eight = "NO"
                        ans_nine = "Yes"
                        ans_ten = "same"
                    else:
                        ans_eight = "NO"
                        ans_nine = "Yes"
                        ans_ten = "shorter"
                    qst_ans_list.append([qst_eight, ans_eight])
                    qst_ans_list.append([qst_nine, ans_nine])
                    qst_ans_list.append([qst_ten, ans_ten])
                    for qst_ans in qst_ans_list_:
                        self.jion_current_dict("3" + str(counter), qst_ans)
                        counter = counter + 1
                    i = i + 1
            self.join_attribute_dict("LocDis", dis_loc_list)

    # Establish a category dictionary. When a major category of landcover appears, only the minor categories should be presented in the subset，self.LC2Subset{LC1:[subset1,subset2,...,subsetn],...,LC2:[subset1,subset2,...,subsetn]}

    # When selecting landcover, the object automatically appears as a subcategory
    def get_traffic_object(self):
        landcover = self.traffic_ans_landcover.currentText()
        self.traffic_ans_object.clear()
        self.traffic_ans_object.addItems(self.TR2ObjList[landcover])

    def get_residential_object(self):
        landcover = self.residential_ans_landcover.currentText()
        self.residential_ans_object.clear()
        self.residential_ans_object.addItems(self.RA2ObjList[landcover])

    def get_current_traffic_landcover(self):
        if self.traffic_ans_landcover.currentText() != "LandCover Class":
            self.current_traffic_landcover = self.traffic_ans_landcover.currentText()

    def get_current_residential_landcover(self):
        if self.residential_ans_landcover.currentText() != "LandCover Class":
            self.current_residential_landcover = (
                self.residential_ans_landcover.currentText()
            )

    def join_traffic(self):
        if self.question_traffic.isChecked() and self.all_Index_Check(3):
            current_traffic_list = []
            i = 1
            while i < len(self.traffic_list):
                current_traffic_list.append(
                    [
                        self.traffic_list[i].currentIndex(),
                        self.traffic_list[i].currentText(),
                    ]
                )
                i = i + 1
            if self.traffic_list[0].currentText() not in self.traffic_dict:
                self.traffic_dict[self.traffic_list[0].currentText()] = []
            if len(self.traffic_dict[self.traffic_list[0].currentText()]) > 0:
                flag = True
                for idx, subset in enumerate(
                    self.traffic_dict[self.traffic_list[0].currentText()]
                ):
                    if subset[0][1] == current_traffic_list[0][1]:
                        self.traffic_dict[self.traffic_list[0].currentText()][
                            idx
                        ] = current_traffic_list
                        flag = False
                        break
                if flag:
                    self.traffic_dict[self.traffic_list[0].currentText()].append(
                        current_traffic_list
                    )
            else:
                self.traffic_dict[self.traffic_list[0].currentText()].append(
                    current_traffic_list
                )
            if len(self.traffic_dict) > 0:
                self.display_anno.clear()
                for k, v in self.traffic_dict.items():
                    if len(v) > 0:
                        for s in v:
                            # Display traffic annotation, Num, Loc, Better
                            self.display_anno.append(
                                "Traffic (%s:%s): Num %s, Loc %s, Better %s"
                                % (k, s[0][1], s[1][1], s[2][1], s[3][1])
                            )
                            # self.display_anno.append('Traffic (%s:%s): Num %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s' % (k, s[0][1], s[1][1], s[2][1], s[3][1], s[4][1], s[5][1], s[6][1], s[7][1]))
            for e in self.traffic_list:
                e.setCurrentIndex(0)
        else:
            self.Warn_Erorr("Some checkboxes are not selected, cannot confirm")

    def join_residential(self):
        if self.question_residential.isChecked() and self.all_Index_Check(4):
            current_residential_list = []
            i = 1
            while i < len(self.residential_list):
                current_residential_list.append(
                    [
                        self.residential_list[i].currentIndex(),
                        self.residential_list[i].currentText(),
                    ]
                )
                i = i + 1
            if self.residential_list[0].currentText() not in self.residential_dict:
                self.residential_dict[self.residential_list[0].currentText()] = []
            if len(self.residential_dict[self.residential_list[0].currentText()]) > 0:
                flag = True
                for idx, subset in enumerate(
                    self.residential_dict[self.residential_list[0].currentText()]
                ):
                    if subset[0][1] == current_residential_list[0][1]:
                        self.residential_dict[self.residential_list[0].currentText()][
                            idx
                        ] = current_residential_list
                        flag = False
                        break
                if flag:
                    self.residential_dict[
                        self.residential_list[0].currentText()
                    ].append(current_residential_list)
            else:
                self.residential_dict[self.residential_list[0].currentText()].append(
                    current_residential_list
                )
            if len(self.residential_dict) > 0:
                self.display_anno.clear()
                for k, v in self.residential_dict.items():
                    if len(v) > 0:
                        for s in v:
                            self.display_anno.append(
                                "Residential (%s:%s): Num %s, Loc %s, Better %s"
                                % (k, s[0][1], s[1][1], s[2][1], s[3][1])
                            )
            for e in self.residential_list:
                e.setCurrentIndex(0)
        else:
            self.Warn_Erorr("Some checkboxes are not selected, cannot confirm")

    # When selecting landcover, the subset automatically appears as a subcategory

    def get_subset(self):
        landcover = self.contain_ans_landcover.currentText()
        self.contain_ans_subset.clear()
        self.contain_ans_subset.addItems(self.LC2SubList[landcover])

    def get_current_landcover(self):
        if self.contain_ans_landcover.currentText() != "LandCover Class":
            self.current_landcover = self.contain_ans_landcover.currentText()

    def join_contain(self):
        if self.question_contain.isChecked() and self.all_Index_Check(2):
            # Store the information to be submitted for contain/presence, [[index,text],[],...], a two-dimensional array, excluding the type of landcover
            current_contain_list = []
            i = 1
            while i < len(
                self.contain_list
            ):  # self.contain_list contains the 9 components of contain/presence
                current_contain_list.append(
                    [
                        self.contain_list[i].currentIndex(),
                        self.contain_list[i].currentText(),
                    ]
                )
                i = i + 1
            # self.contain_dict stores each landcover corresponding to several 8 components {landcover1:[[1,2,3,...,8],[1,2,3,...,8]],landcover2:[[1,2,3,...,8],[1,2,3,...,8]],...}
            if self.contain_list[0].currentText() not in self.contain_dict:
                self.contain_dict[self.contain_list[0].currentText()] = []
            if len(self.contain_dict[self.contain_list[0].currentText()]) > 0:
                flag = True
                for idx, subset in enumerate(
                    self.contain_dict[self.contain_list[0].currentText()]
                ):
                    if subset[0][1] == current_contain_list[0][1]:
                        self.contain_dict[self.contain_list[0].currentText()][
                            idx
                        ] = current_contain_list
                        flag = False
                        break
                if flag:
                    self.contain_dict[self.contain_list[0].currentText()].append(
                        current_contain_list
                    )
            else:
                self.contain_dict[self.contain_list[0].currentText()].append(
                    current_contain_list
                )

            # Display the attribute information of the submitted objects
            if len(self.contain_dict) > 0:
                self.display_anno.clear()
                for k, v in self.contain_dict.items():
                    if len(v) > 0:
                        for s in v:
                            self.display_anno.append(
                                "Con&&Pre (%s:%s): Num %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s"
                                % (
                                    k,
                                    s[0][1],
                                    s[1][1],
                                    s[2][1],
                                    s[3][1],
                                    s[4][1],
                                    s[5][1],
                                    s[6][1],
                                    s[7][1],
                                )
                            )

            for e in self.contain_list:
                e.setCurrentIndex(0)
        else:
            self.Warn_Erorr("Some checkboxes are not selected, cannot confirm")

    # If landcover and subset have been previously annotated, and the landcover and subset are selected again, the remaining properties will be automatically filled

    def show_pre_contain_attribution(self):
        if self.flag:
            if (
                self.contain_ans_landcover.currentText() != "LandCover Class"
                and self.contain_ans_subset.currentText() != "Subset Class"
            ):
                landcover = self.contain_ans_landcover.currentText()
                subset = self.contain_ans_subset.currentText()
                subset_attribution_list = [
                    self.contain_ans_number,
                    self.contain_ans_location,
                    self.contain_ans_shape,
                    self.contain_ans_Area,
                    self.contain_ans_length,
                    self.contain_ans_distribution,
                    self.contain_ans_better,
                ]

                if landcover in self.contain_dict:
                    landcover_attribution = self.contain_dict[landcover]
                    for sub_attribution in landcover_attribution:
                        if sub_attribution[0][1] == subset:
                            for idx, attributions in enumerate(sub_attribution[1:]):
                                subset_attribution_list[idx].setCurrentIndex(
                                    attributions[0]
                                )

    # If landcover and object have been previously annotated, and the landcover and object are selected again, the remaining properties will be automatically filled
    def show_pre_traffic_attribution(self):
        if self.flag:
            if (
                self.traffic_ans_landcover.currentText() != "LandCover Class"
                and self.traffic_ans_object.currentText() != "Object Class"
            ):
                landcover = self.traffic_ans_landcover.currentText()
                object = self.traffic_ans_object.currentText()
                object_attribution_list = [
                    self.traffic_ans_number,
                    self.traffic_ans_location,
                    self.traffic_ans_better,
                ]

                if landcover in self.traffic_dict:
                    landcover_attribution = self.traffic_dict[landcover]
                    for sub_attribution in landcover_attribution:
                        if sub_attribution[0][1] == object:
                            for idx, attributions in enumerate(sub_attribution[1:]):
                                object_attribution_list[idx].setCurrentIndex(
                                    attributions[0]
                                )

    def show_pre_residential_attribution(self):
        if self.flag:
            if (
                self.residential_ans_landcover.currentText() != "LandCover Class"
                and self.residential_ans_object.currentText() != "Object Class"
            ):
                landcover = self.residential_ans_landcover.currentText()
                object = self.residential_ans_object.currentText()
                object_attribution_list = [
                    self.residential_ans_number,
                    self.residential_ans_location,
                    self.residential_ans_better,
                ]

                if landcover in self.residential_dict:
                    landcover_attribution = self.residential_dict[landcover]
                    for sub_attribution in landcover_attribution:
                        if sub_attribution[0][1] == object:
                            for idx, attributions in enumerate(sub_attribution[1:]):
                                object_attribution_list[idx].setCurrentIndex(
                                    attributions[0]
                                )

    def del_contain(self):
        if not self.question_contain.isChecked():
            self.del_annotation(q="4")
            self.del_attribute_dict("PresContain")

    def del_traffic(self):
        if not self.question_traffic.isChecked():
            self.del_annotation(q="6")
            self.del_attribute_dict("Traffic")

    def del_residential(self):
        if not self.question_residential.isChecked():
            self.del_annotation(q="7")
            self.del_attribute_dict("Residential")

    def generate_contain(self):
        qst_list = []  # Used to store all the question pairs
        counter = 1  # Used to number the questions
        # self.contain_dict stores each landcover corresponding to several 8 components {landcover1:[[1,2,3,...,8],[1,2,3,...,9]],landcover2:}
        if self.question_contain.isChecked() and len(self.contain_dict) > 0:
            land_cover_list = []  # [landcover1,landcover2,...]
            # {landcover1:[subset1,subset2,...],landcover2:[subset1,...]}
            landcover_dict = {}
            # {landcover1:[number(index之和),area(index之和),kinds],landcover2:[number,area,kinds],...}
            landcover_atri_dict = {}
            # {subset1:[[index,text],[index,text],...],subset2:[[index,text],[index,text],...],...}
            subset_dict = {}
            subset_list = []  # [subset1,subset2,...]
            for k, v in self.contain_dict.items():
                land_cover_list.append(k)
                temp = []  # Temporary storage

                for subset in v:
                    temp.append(subset[0][1])
                    subset_dict[subset[0][1]] = subset[1:]
                    subset_list.append(subset[0][1])
                landcover_dict[k] = temp

            # Establish an attribute dictionary for landcover, including the quantity, area, and type of each landcover category
            # Collect the quantity, area, and number of types of landcover levels landcover_atri_dict:{landcover1:[num,area,kinds],landcover2:[num,area,kinds],...}
            for nu in range(len(land_cover_list)):
                landcover = land_cover_list[nu]
                num_s = 0
                area_s = 0

                if len(landcover_dict[landcover]) > 0:
                    for n_ in range(len(landcover_dict[landcover])):
                        sub = landcover_dict[landcover][n_]
                        num_s = num_s + subset_dict[sub][0][0]
                        area_s = area_s + subset_dict[sub][4][0]
                    landcover_atri_dict[landcover] = [
                        num_s,
                        area_s,
                        len(landcover_dict[landcover]),
                    ]

            # For the landcover Q&A section, ask questions about the overall situation of landcover
            qst_1 = "How many land cover classes are totally in this picture ?"
            ans_1 = len(land_cover_list)
            qst_list.append([qst_1, ans_1])

            # Sort landcover by quantity and area
            # sort_land_cover_num=sorted(landcover_atri_dict,key=lambda k:landcover_atri_dict[k][0],reverse=True)  # 数量从大到小的landcover list

            sort_land_cover_area = sorted(
                landcover_atri_dict,
                key=lambda k: landcover_atri_dict[k][1],
                reverse=True,
            )  # Area from largest to smallest landcover list

            # Keep only one landcover in the landcover list, and ask which landcover is the largest

            # Area and quantity sorting, then ask (maximum and minimum)
            if len(sort_land_cover_area) > 1:
                if (
                    landcover_atri_dict[sort_land_cover_area[0]]
                    != landcover_atri_dict[sort_land_cover_area[1]]
                ):
                    qst_2 = "Which land cover classes is the largest in this picture ?"
                    ans_2 = sort_land_cover_area[0]
                    qst_list.append([qst_2, ans_2])
                elif (
                    len(sort_land_cover_area) == 2
                    or landcover_atri_dict[sort_land_cover_area[1]]
                    != landcover_atri_dict[sort_land_cover_area[2]]
                ):
                    qst_2 = (
                        "In addition to the %ss, Which land cover classes is the largest in this picture ?"
                        % (sort_land_cover_area[0])
                    )
                    ans_2 = sort_land_cover_area[1]
                    qst_ = (
                        "In addition to the %ss, Which land cover classes is the largest in this picture ?"
                        % (sort_land_cover_area[1])
                    )
                    ans_ = sort_land_cover_area[0]
                    qst_list.append([qst_2, ans_2])
                    qst_list.append([qst_, ans_])

                if (
                    landcover_atri_dict[sort_land_cover_area[-1]]
                    != landcover_atri_dict[sort_land_cover_area[-2]]
                ):
                    qst_2 = "Which land cover classes is the smallest in this picture ?"
                    ans_2 = sort_land_cover_area[-1]
                    qst_list.append([qst_2, ans_2])
                elif (
                    len(sort_land_cover_area) > 2
                    and landcover_atri_dict[sort_land_cover_area[-2]]
                    != landcover_atri_dict[sort_land_cover_area[-3]]
                ):
                    qst_2 = (
                        "In addition to the %ss, Which land cover classes is the smallest in this picture ?"
                        % (sort_land_cover_area[-1])
                    )
                    ans_2 = sort_land_cover_area[-2]
                    qst_ = (
                        "In addition to the %ss, Which land cover classes is the largetst in this picture ?"
                        % (sort_land_cover_area[-2])
                    )
                    ans_ = sort_land_cover_area[-1]
                    qst_list.append([qst_2, ans_2])
                    qst_list.append([qst_, ans_])

            # Make the number of answers 1 landcover
            if len(land_cover_list) == 1:
                qst_2 = "What land cover classes are in this picture ?"
                ans_2 = land_cover_list[0]
                qst_list.append([qst_2, ans_2])

            elif len(land_cover_list) == 2:
                for n in range(len(land_cover_list)):
                    qst_3 = (
                        "In addition to the %ss, Which cover classes is in this picture ?"
                        % (land_cover_list[n])
                    )
                    ans_3 = land_cover_list[(n + 1) % 2]
                    qst_list.append([qst_3, ans_3])
            elif len(land_cover_list) == 3:
                for n in range(len(land_cover_list)):
                    qst_3 = (
                        "In addition to the %ss, Which cover classes in this picture ?"
                        % (
                            " and ".join(
                                [
                                    land
                                    for land in land_cover_list
                                    if land != land_cover_list[n]
                                ]
                            )
                        )
                    )
                    ans_3 = land_cover_list[n]
                    qst_list.append([qst_3, ans_3])
            else:
                for n in range(len(land_cover_list)):
                    temp_ = [
                        land for land in land_cover_list if land != land_cover_list[n]
                    ]  # Exclude the answer landcover后的 landcoverlist
                    qst_3 = (
                        "In addition to the %ss, Which cover classes is in this picture ?"
                        % (", ".join(temp_[:-1]) + " and %s" % temp_[-1])
                    )
                    ans_3 = land_cover_list[n]
                    qst_list.append([qst_3, ans_3])

            non_land_cover_list = [
                land for land in self.landcover_list if land not in land_cover_list
            ]  # The landcover type that did not appear
            random.shuffle(non_land_cover_list)
            if len(land_cover_list) <= len(non_land_cover_list):
                num = len(land_cover_list)
            else:
                num = len(non_land_cover_list)
            for i in range(num):
                qst_list.append(
                    [
                        "Is there some a %s present in this picture ?"
                        % land_cover_list[i],
                        "Yes",
                    ]
                )
                qst_list.append(
                    [
                        "Is there some a %s present in this picture ?"
                        % non_land_cover_list[i],
                        "No",
                    ]
                )

            # Traverse the landcovers that appear and ask questions for each landcover
            for n in range(len(land_cover_list)):
                landcover = land_cover_list[n]
                qst_list.append(
                    [
                        "How many %s classes are in this picture ?" % landcover,
                        "%d" % len(landcover_dict[landcover]),
                    ]
                )
                # For the subset Q&A section, ask questions about the types of subsets  
                if len(landcover_dict[landcover]) == 1:
                    qst_list.append(
                        [
                            "What %s classes are in this picture ?" % landcover,
                            landcover_dict[landcover][0],
                        ]
                    )
                elif len(landcover_dict[landcover]) == 2:
                    qst_list.append(
                        [
                            "In addition to the %ss, which %s classes does this picture contain ?"
                            % (landcover_dict[landcover][0], landcover),
                            landcover_dict[landcover][1],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss, which %s classes does this picture contain ?"
                            % (landcover_dict[landcover][1], landcover),
                            landcover_dict[landcover][0],
                        ]
                    )
                elif len(landcover_dict[landcover]) == 3:
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][0],
                                landcover_dict[landcover][1],
                                landcover,
                            ),
                            landcover_dict[landcover][2],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][0],
                                landcover_dict[landcover][2],
                                landcover,
                            ),
                            landcover_dict[landcover][1],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][1],
                                landcover_dict[landcover][2],
                                landcover,
                            ),
                            landcover_dict[landcover][0],
                        ]
                    )
                else:
                    pass  # For the comparison part of landcover, ask questions in the same way as the overall landcover inquiry, asking questions in the form of "except what", and reducing the number of answers to 1
                # For the comparison part of landcover
                m = n + 1
                while m < len(land_cover_list):
                    landcover_ = land_cover_list[m]
                    # Compare quantity
                    if (
                        landcover_atri_dict[landcover][0]
                        > landcover_atri_dict[landcover_][0]
                    ):
                        ans = "More"
                        ans_a = "Less"
                        ans_ = "Yes"
                        ans__ = "No"
                    elif (
                        landcover_atri_dict[landcover][0]
                        == landcover_atri_dict[landcover_][0]
                    ):
                        ans = "Almost Same"
                        ans_a = "Almost Same"
                        ans_ = "No"
                        ans__ = "No"
                    else:
                        ans = "Less"
                        ans_a = "More"
                        ans_ = "No"
                        ans__ = "Yes"
                    qst_list.append(
                        [
                            "How is the number of the %ss in this picture compared to the %ss ?"
                            % (landcover, landcover_),
                            ans,
                        ]
                    )
                    qst_list.append(
                        [
                            "How is the number of the %ss in this picture compared to the %ss ?"
                            % (landcover_, landcover),
                            ans_a,
                        ]
                    )
                    qst_list.append(
                        [
                            "Are there more %ss than %ss in this picture ?"
                            % (landcover, landcover_),
                            ans_,
                        ]
                    )
                    qst_list.append(
                        [
                            "Are there more %ss than %ss in this picture ?"
                            % (landcover_, landcover),
                            ans__,
                        ]
                    )

                    # Compare area
                    if (
                        landcover_atri_dict[landcover][1]
                        > landcover_atri_dict[landcover_][1]
                    ):
                        ans = "Larger"
                        ans_a = "Smaller"
                        ans_ = "Yes"
                        ans__ = "No"
                    elif (
                        landcover_atri_dict[landcover][1]
                        == landcover_atri_dict[landcover_][1]
                    ):
                        ans = "Almost Same"
                        ans_a = "Almost Same"
                        ans_ = "No"
                        ans__ = "No"
                    else:
                        ans = "Smaller"
                        ans_a = "Larger"
                        ans_ = "No"
                        ans__ = "Yes"
                    qst_list.append(
                        [
                            "How is the area of the %ss in this picture compared to the %ss ?"
                            % (landcover, landcover_),
                            ans,
                        ]
                    )
                    qst_list.append(
                        [
                            "How is the area of the %ss in this picture compared to the %ss ?"
                            % (landcover_, landcover),
                            ans_a,
                        ]
                    )
                    qst_list.append(
                        [
                            "Is the area of the %ss in this picture greater than the area of the %ss ?"
                            % (landcover, landcover_),
                            ans_,
                        ]
                    )
                    qst_list.append(
                        [
                            "Is the area of the %ss in this picture greater than the area of the %ss ?"
                            % (landcover_, landcover),
                            ans__,
                        ]
                    )
                    # Compare the number of types
                    if (
                        landcover_atri_dict[landcover][2]
                        > landcover_atri_dict[landcover_][2]
                    ):
                        ans_ = "Yes"
                        ans__ = "No"
                    else:
                        ans_ = "No"
                        ans__ = "Yes"
                    qst_list.append(
                        [
                            "Is there more categories of %ss in this picture than %ss ?"
                            % (landcover, landcover_),
                            ans_,
                        ]
                    )
                    qst_list.append(
                        [
                            "Is there more categories of %ss in this picture than %ss ?"
                            % (landcover_, landcover),
                            ans__,
                        ]
                    )

                    m = m + 1

            # The presence of subset
            non_subset_list = [
                subset
                for subset in self.subset_list
                if subset not in landcover_dict[landcover]
            ] + [
                "orchard",
                "military area",
                "airport",
                "vineyard",
                "volcano",
            ]  # The subset list that did not appear
            random.shuffle(non_subset_list)
            if len(subset_list) <= len(non_subset_list):
                num = len(subset_list)
            else:
                num = len(non_subset_list)
            for i in range(num):
                qst_list.append(
                    ["Are there some %ss present ?" % subset_list[n], "Yes"]
                )
                qst_list.append(
                    ["Are there some %ss present ?" % non_subset_list[n], "No"]
                )
                qst_list.append(
                    ["How many %ss are in this picture ?" % non_subset_list[n], "0"]
                )

            # Other questions针对subset

            for n in range(len(subset_list)):
                subset = subset_list[n]
                # number question
                # If the quantity attribute exists, farmland and thickets take a separate number
                if subset_dict[subset][0][1] != "none":
                    if subset == "farmlands" or subset == "thickets":
                        qst_list.append(
                            [
                                "How many separated %ss are in this picture ?" % subset,
                                subset_dict[subset][0][1],
                            ]
                        )
                    else:
                        qst_list.append(
                            [
                                "How many %ss are in this picture ?" % subset,
                                subset_dict[subset][0][1],
                            ]
                        )

                # location question
                if subset_dict[subset][1][1] != "none":
                    qst_list.append(
                        [
                            "Where is the most part of %ss ?" % subset,
                            subset_dict[subset][1][1],
                        ]
                    )
                # shape question
                if subset_dict[subset][2][1] != "none":
                    qst_list.append(
                        [
                            "What is the shape of the largest %ss ?" % subset,
                            subset_dict[subset][2][1],
                        ]
                    )
                # Area question
                if subset_dict[subset][3][1] != "none":
                    qst_list.append(
                        [
                            "What is the area covered by %ss ?" % subset,
                            subset_dict[subset][3][1],
                        ]
                    )
                # length question
                if subset_dict[subset][4][1] != "none":
                    qst_list.append(
                        [
                            "What is the length of the longest %ss ?" % subset,
                            subset_dict[subset][4][1],
                        ]
                    )
                # Distribution question
                if subset_dict[subset][5][1] != "none":
                    qst_list.append(
                        [
                            "How are the most part of the %ss distributed ?" % subset,
                            subset_dict[subset][5][1],
                        ]
                    )
                # Better question
                if subset_dict[subset][6][1] != "none":
                    qst_list.append(
                        [
                            "Which picture has %ss in it is better ?" % subset,
                            subset_dict[subset][6][1],
                        ]
                    )

                # Comparison question, compare subsets.
                m = n + 1
                while m < len(subset_list):
                    subset_ = subset_list[m]
                    # Compare quantity
                    if (
                        subset_dict[subset][0][1] != "none"
                        and subset_dict[subset_][0][1] != "none"
                    ):
                        if subset_dict[subset][0][0] > subset_dict[subset_][0][0]:
                            ans = "More"
                            ans_a = "Less"
                            ans_ = "Yes"
                            ans__ = "No"
                        elif subset_dict[subset][0][0] == subset_dict[subset_][0][0]:
                            ans = "Almost Same"
                            ans_a = "Almost Same"
                            ans_ = "No"
                            ans__ = "No"
                        else:
                            ans = "Less"
                            ans_a = "More"
                            ans_ = "No"
                            ans__ = "Yes"
                        qst_list.append(
                            [
                                "How about the number of %ss in this picture compared to %ss ?"
                                % (subset, subset_),
                                ans,
                            ]
                        )
                        qst_list.append(
                            [
                                "How about the number of %ss in this picture compared to %ss ?"
                                % (subset_, subset),
                                ans_a,
                            ]
                        )
                        qst_list.append(
                            [
                                "Are there more %ss than %ss in this picture ?"
                                % (subset, subset_),
                                ans_,
                            ]
                        )
                        qst_list.append(
                            [
                                "Are there more %ss than %ss in this picture ?"
                                % (subset_, subset),
                                ans__,
                            ]
                        )
                    # Compare area
                    if (
                        subset_dict[subset][3][1] != "none"
                        and subset_dict[subset_][3][1] != "none"
                    ):
                        if subset_dict[subset][3][0] > subset_dict[subset_][3][0]:
                            ans = "Larger"
                            ans_a = "Smaller"
                            ans_ = "Yes"
                            ans__ = "No"
                        elif subset_dict[subset][3][0] == subset_dict[subset_][3][0]:
                            ans = "Almost Same"
                            ans_a = "Almost Same"
                            ans_ = "No"
                            ans__ = "No"
                        else:
                            ans = "Smaller"
                            ans_a = "Larger"
                            ans_ = "No"
                            ans__ = "Yes"
                        qst_list.append(
                            [
                                "How about the area of %ss in this picture compared to %ss ?"
                                % (subset, subset_),
                                ans,
                            ]
                        )
                        qst_list.append(
                            [
                                "How about the area of %ss in this picture compared to %ss ?"
                                % (subset_, subset),
                                ans_a,
                            ]
                        )
                        qst_list.append(
                            [
                                "Whether the area of %ss in this picture is greater than the area of %ss ?"
                                % (subset, subset_),
                                ans_,
                            ]
                        )
                        qst_list.append(
                            [
                                "Whether the area of %ss in this picture is greater than the area of %ss ?"
                                % (subset_, subset),
                                ans__,
                            ]
                        )
                    # Compare length
                    if (
                        subset_dict[subset][4][1] != "none"
                        and subset_dict[subset_][4][1] != "none"
                    ):
                        if subset_dict[subset][4][0] > subset_dict[subset_][4][0]:
                            ans = "Longer"
                            ans_a = "Shorter"
                            ans_ = "Yes"
                            ans__ = "No"
                        elif subset_dict[subset][4][0] == subset_dict[subset_][4][0]:
                            ans = "almost same"
                            ans_a = "almost same"
                            ans_ = "No"
                            ans__ = "No"
                        else:
                            ans = "Shorter"
                            ans_a = "Longer"
                            ans_ = "No"
                            ans__ = "Yes"
                        qst_list.append(
                            [
                                "How about the length of %ss in this picture compared to %ss ?"
                                % (subset, subset_),
                                ans,
                            ]
                        )
                        qst_list.append(
                            [
                                "How about the length of %ss in this picture compared to %ss ?"
                                % (subset_, subset),
                                ans_a,
                            ]
                        )
                        qst_list.append(
                            [
                                "Whether the length of %ss in this picture is greater than the area of %ss ?"
                                % (subset, subset_),
                                ans_,
                            ]
                        )
                        qst_list.append(
                            [
                                "Whether the length of %ss in this picture is greater than the area of %ss ?"
                                % (subset_, subset),
                                ans__,
                            ]
                        )
                    m = m + 1

            for qst_ans in qst_list:
                self.jion_current_dict("4%d" % counter, qst_ans)
                counter = counter + 1
            self.join_attribute_dict("PresContain", self.contain_dict)

    # Generate traffic questions
    def generate_traffic(self):
        qst_list = []
        counter = 1  # Used to number the questions
        if self.question_traffic.isChecked() and len(self.traffic_dict) > 0:
            land_cover_list = []
            landcover_dict = {}
            landcover_atri_dict = {}
            object_dict = {}
            object_list = []
            for k, v in self.traffic_dict.items():
                land_cover_list.append(k)
                temp = []
                for object in v:
                    temp.append(object[0][1])
                    object_dict[object[0][1]] = object[1:]
                    object_list.append(object[0][1])
                landcover_dict[k] = temp

            for n in range(len(land_cover_list)):
                landcover = land_cover_list[n]
                qst_list.append(
                    [
                        "How many %s classes are in this picture ?" % landcover,
                        "%d" % len(landcover_dict[landcover]),
                    ]
                )
                if len(landcover_dict[landcover]) == 1:
                    qst_list.append(
                        [
                            "What %s classes are in this picture ?" % landcover,
                            landcover_dict[landcover][0],
                        ]
                    )
                elif len(landcover_dict[landcover]) == 2:
                    qst_list.append(
                        [
                            "In addition to the %ss, which %s classes does this picture contain ?"
                            % (landcover_dict[landcover][0], landcover),
                            landcover_dict[landcover][1],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss, which %s classes does this picture contain ?"
                            % (landcover_dict[landcover][1], landcover),
                            landcover_dict[landcover][0],
                        ]
                    )
                elif len(landcover_dict[landcover]) == 3:
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][0],
                                landcover_dict[landcover][1],
                                landcover,
                            ),
                            landcover_dict[landcover][2],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][0],
                                landcover_dict[landcover][2],
                                landcover,
                            ),
                            landcover_dict[landcover][1],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][1],
                                landcover_dict[landcover][2],
                                landcover,
                            ),
                            landcover_dict[landcover][0],
                        ]
                    )
                else:
                    pass
            # The presence of object
            non_object_list = [
                object
                for object in self.traffic_object_list
                if object not in object_list
            ]
            random.shuffle(non_object_list)
            if len(object_list) <= len(non_object_list):
                num = len(object_list)
            else:
                num = len(non_object_list)
            for i in range(num):
                qst_list.append(
                    ["Are there some %ss present ?" % object_list[n], "Yes"]
                )
                qst_list.append(
                    ["Are there some %ss present ?" % non_object_list[n], "No"]
                )
                qst_list.append(
                    ["How many %ss are in this picture ?" % non_object_list[n], "0"]
                )

            # Other questions for object
            for n in range(len(object_list)):
                object = object_list[n]
                # number question
                if object_dict[object][0][1] != "none":
                    qst_list.append(
                        [
                            "How many %ss are in this picture ?" % object,
                            object_dict[object][0][1],
                        ]
                    )
                # location question
                if object_dict[object][1][1] != "none":
                    qst_list.append(
                        [
                            "Where is the most part of %ss ?" % object,
                            object_dict[object][1][1],
                        ]
                    )
                # better question
                if object_dict[object][2][1] != "none":
                    qst_list.append(
                        [
                            "Which picture has %ss in it is better ?" % object,
                            object_dict[object][2][1],
                        ]
                    )

            for qst_ans in qst_list:
                self.jion_current_dict("6%d" % counter, qst_ans)
                counter = counter + 1
            self.join_attribute_dict("Traffic", self.traffic_dict)

    # Generate residential questions
    def generate_residential(self):
        qst_list = []
        counter = 1
        if self.question_residential.isChecked() and len(self.residential_dict) > 0:
            land_cover_list = []
            landcover_dict = {}
            landcover_atri_dict = {}
            object_dict = {}
            object_list = []
            for k, v in self.residential_dict.items():
                land_cover_list.append(k)
                temp = []
                for object in v:
                    temp.append(object[0][1])
                    object_dict[object[0][1]] = object[1:]
                    object_list.append(object[0][1])
                landcover_dict[k] = temp

            for n in range(len(land_cover_list)):
                landcover = land_cover_list[n]
                qst_list.append(
                    [
                        "How many %s classes are in this picture ?" % landcover,
                        "%d" % len(landcover_dict[landcover]),
                    ]
                )
                if len(landcover_dict[landcover]) == 1:
                    qst_list.append(
                        [
                            "What %s classes are in this picture ?" % landcover,
                            landcover_dict[landcover][0],
                        ]
                    )
                elif len(landcover_dict[landcover]) == 2:
                    qst_list.append(
                        [
                            "In addition to the %ss, which %s classes does this picture contain ?"
                            % (landcover_dict[landcover][0], landcover),
                            landcover_dict[landcover][1],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss, which %s classes does this picture contain ?"
                            % (landcover_dict[landcover][1], landcover),
                            landcover_dict[landcover][0],
                        ]
                    )
                elif len(landcover_dict[landcover]) == 3:
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][0],
                                landcover_dict[landcover][1],
                                landcover,
                            ),
                            landcover_dict[landcover][2],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][0],
                                landcover_dict[landcover][2],
                                landcover,
                            ),
                            landcover_dict[landcover][1],
                        ]
                    )
                    qst_list.append(
                        [
                            "In addition to the %ss and %ss, which %s classes does this picture contain ?"
                            % (
                                landcover_dict[landcover][1],
                                landcover_dict[landcover][2],
                                landcover,
                            ),
                            landcover_dict[landcover][0],
                        ]
                    )
                else:
                    pass
            # The presence of object
            non_object_list = [
                object
                for object in self.residential_object_list
                if object not in object_list
            ]
            random.shuffle(non_object_list)
            if len(object_list) <= len(non_object_list):
                num = len(object_list)
            else:
                num = len(non_object_list)
            for i in range(num):
                qst_list.append(
                    ["Are there some %ss present ?" % object_list[n], "Yes"]
                )
                qst_list.append(
                    ["Are there some %ss present ?" % non_object_list[n], "No"]
                )
                qst_list.append(
                    ["How many %ss are in this picture ?" % non_object_list[n], "0"]
                )

            # Other questions for object
            for n in range(len(object_list)):
                object = object_list[n]
                # number question
                if object_dict[object][0][1] != "none":
                    qst_list.append(
                        [
                            "How many %ss are in this picture ?" % object,
                            object_dict[object][0][1],
                        ]
                    )
                # location question
                if object_dict[object][1][1] != "none":
                    qst_list.append(
                        [
                            "Where is the most part of %ss ?" % object,
                            object_dict[object][1][1],
                        ]
                    )
                # better question
                if object_dict[object][2][1] != "none":
                    qst_list.append(
                        [
                            "Which picture has %ss in it is better ?" % object,
                            object_dict[object][2][1],
                        ]
                    )

            for qst_ans in qst_list:
                self.jion_current_dict("6%d" % counter, qst_ans)
                counter = counter + 1
            self.join_attribute_dict("Residential", self.residential_dict)

    # Display the attribute information of the annotation

    def display_attribution_(self):
        if len(self.current_attri_dict) > 0:
            self.display_anno.clear()
            for k, v in self.current_attri_dict.items():
                if (
                    k == "match"
                    or k == "theme"
                    or k == "urban"
                    or k == "Deduce_one"
                    or k == "Deduce_two"
                    or k == "mist"
                    or k == "night"
                    or k == "agricultural_road"
                    or k == "water_source"
                    or k == "logistics_facility"
                    or k == "construction_scale"
                    or k == "industrial_location"
                    or k == "UAV_height"
                    or k == "UAV_angle"
                ):
                    self.display_anno.append("%s:%s" % (k, v))
                elif k == "LocDis":
                    if len(v) > 0:
                        for s in v:
                            self.display_anno.append(
                                "%s:(%s,%s):Loc %s, Dis %s"
                                % (k, s[0], s[1], s[3], s[4])
                            )
                elif k == "PresContain":
                    for m, l in v.items():
                        if len(l) > 0:
                            for s in l:
                                self.display_anno.append(
                                    "%s:\n(%s:%s):\nnum %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s"
                                    % (
                                        k,
                                        m,
                                        s[0],
                                        s[1],
                                        s[2],
                                        s[3],
                                        s[4],
                                        s[5],
                                        s[6],
                                        s[7],
                                    )
                                )
                elif k == "Traffic" or k == "Residential":
                    for m, l in v.items():
                        if len(l) > 0:
                            for s in l:
                                self.display_anno.append(
                                    "%s:\n(%s:%s):\nnum %s, Loc %s, Better %s"
                                    % (k, m, s[0], s[1], s[2], s[3])
                                )

    # #On February 9, 2024, it was revised. To correct "PreContain", only "preContain" was displayed in the annotation
    # def display_attribution(self):
    #     if len(self.current_attri_dict)>0:
    #         self.display_anno.clear()
    #         for k,v in self.current_attri_dict.items():
    #             if k=='PresContain':
    #                 for m,l in v.items():
    #                     if len(l)>0:
    #                         for s in l:
    #                             self.display_anno.append('%s:\n(%s:%s):\nnum %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s'%(k,m,s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]))

    # On February 9, 2024, it was revised. To correct "PreContain", only "preContain" was displayed in the annotation
    def display_attribution(self):
        # if len(self.current_attri_dict) > 0:
        #     self.display_anno.clear()
        #     for k, v in self.current_attri_dict.items():
        #         if k == 'match' or k == 'theme' or k == 'urban' or k == 'Deduce_one' or k == 'Deduce_two' or k == 'mist':
        #             self.display_anno.append('%s:%s' % (k, v))
        #         elif k == 'LocDis':
        #             if len(v) > 0:
        #                 for s in v:
        #                     self.display_anno.append(
        #                         '%s:(%s,%s):Loc %s, Dis %s' % (k, s[0], s[1], s[3], s[4]))
        #         elif k == 'PresContain':
        #             for m, l in v.items():
        #                 if len(l) > 0:
        #                     for s in l:
        #                         self.display_anno.append('%s:\n(%s:%s):\nnum %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s' % (
        #                             k, m, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))
        if len(self.current_attri_dict) > 0:
            self.display_anno.clear()
            for k, v in self.current_attri_dict.items():
                if (
                    k == "match"
                    or k == "theme"
                    or k == "urban"
                    or k == "Deduce_one"
                    or k == "Deduce_two"
                    or k == "mist"
                    or k == "night"
                    or k == "agricultural_road"
                    or k == "water_source"
                    or k == "logistics_facility"
                    or k == "construction_scale"
                    or k == "industrial_location"
                    or k == "UAV_height"
                    or k == "UAV_angle"
                ):
                    self.display_anno.append("%s:%s" % (k, v))
                elif k == "LocDis":
                    if len(v) > 0:
                        for s in v:
                            self.display_anno.append(
                                "%s:(%s,%s):Loc %s, Dis %s"
                                % (k, s[0], s[1], s[3], s[4])
                            )
                elif k == "PresContain":
                    for m, l in v.items():
                        if len(l) > 0:
                            for s in l:
                                self.display_anno.append(
                                    "%s:\n(%s:%s):\nnum %s, Loc %s, Shape %s, Area %s, Len %s, Distri %s, Better %s"
                                    % (
                                        k,
                                        m,
                                        s[0],
                                        s[1],
                                        s[2],
                                        s[3],
                                        s[4],
                                        s[5],
                                        s[6],
                                        s[7],
                                    )
                                )
                elif k == "Traffic" or k == "Residential":
                    for m, l in v.items():
                        if len(l) > 0:
                            for s in l:
                                self.display_anno.append(
                                    "%s:\n(%s:%s):\nnum %s, Loc %s, Better %s"
                                    % (k, m, s[0], s[1], s[2], s[3])
                                )

    # Use cv2 to measure distance and area
    def get_distance(self):
        path = self.img_paths_rgb[self.counter]
        image = img(path)
        image.get_dis()

    def get_area(self):
        path = self.img_paths_rgb[self.counter]
        image = img(path)
        image.get_area()

    # Select the address to save the annotation file
    def pick_save(self):

        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        if folder_path:
            self.selected_folder_label_save.setText(folder_path)
            self.save_folder = folder_path
            anno_file_path = os.path.join(folder_path, "annotation_dict.json")
            attri_file_path = os.path.join(folder_path, "attribution_dict.json")

            # If the annotation file already exists, load it and locate the next image that has been annotated
            if os.path.exists(anno_file_path) and os.path.exists(attri_file_path):
                with open(anno_file_path, "r") as f:
                    annotation_load = json.load(f)
                with open(attri_file_path, "r") as f:
                    attribution_load = json.load(f)
                # if len(annotation_load)>0 and (len(annotation_load)+1)==len(attribution_load) :
                if len(attribution_load) > 0:

                    self.annotation_dict = annotation_load
                    self.attribution_dict = attribution_load
                    if "counter" in self.attribution_dict:
                        self.counter = self.attribution_dict["counter"] % self.num_rgb
                    else:
                        self.counter = 0
            else:
                self.counter = 0  # There is no already annotated file, do nothing

            # Update the current buffer and display the image and related information
            self.update_current_variable()
            self.display_rgb()
            self.display_tir()
            self.display_attribution()
            # self.display_current_annotation()

    # Select the location to save the rgb file

    def pick_new_rgb(self):

        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        if folder_path:
            self.selected_folder_label_rgb.setText(folder_path)
            self.selected_folder_rgb = folder_path
            if len(self.tir_name) > 0:
                # This step needs to be modified according to different image naming conventions
                self.rgb_name = self.tir_name
            else:
                _, self.rgb_name = get_img_paths(
                    self.selected_folder_rgb
                )  # Get the path of the image file, return a list
            self.rgb_name = [name for name in self.rgb_name if "tir" not in name]
            self.img_paths_rgb = [
                os.path.join(self.selected_folder_rgb, name) for name in self.rgb_name
            ]  # Get the path of the image file, return a list
            self.num_rgb = len(self.img_paths_rgb)

    def pick_new_tir(self):
        """
        shows a dialog to choose folder with images to label
        """
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        if folder_path:
            self.selected_folder_label_tir.setText(folder_path)
            self.selected_folder_tir = folder_path
            # Ensure the order of images in the folder is consistent
            if len(self.rgb_name) > 0:
                self.tir_name = self.rgb_name
            else:
                _, self.tir_name = get_img_paths(
                    self.selected_folder_tir
                )  # Get the path of the image file, return a list
            self.tir_name = [name for name in self.tir_name if "rgb" not in name]
            self.img_paths_tir = [
                os.path.join(self.selected_folder_tir, name) for name in self.tir_name
            ]  # Get the path of the image file, return a list
            self.num_tir = len(self.img_paths_tir)

    # Display rgb image and related information

    def display_rgb(self):
        if (
            os.path.exists(self.selected_folder_rgb)
            and len(self.img_paths_rgb) > self.counter
        ):
            self.set_image_rgb(self.img_paths_rgb[self.counter])  # Display the first image
            self.image_box_rgb.setGeometry(
                20, 145, self.img_panel_width, self.img_panel_height
            )
            self.image_box_rgb.setAlignment(Qt.AlignTop)  #
            # Display the address of the current image, set it to be copyable
            self.rgb_name_label.setGeometry(20, 120, 400, 20)
            self.rgb_name_label.setStyleSheet("font-weight:bold;font-size:14px")
            self.rgb_name_label.setText("Current RGB: " + self.rgb_name[self.counter])
            # Display progress
            self.progress_bar_rgb.setGeometry(350, 120, 100, 20)
            self.progress_bar_rgb.setStyleSheet("font-weight:bold;font-size:14px")
            self.progress_bar_rgb.setText(f"{self.counter+1} of {self.num_rgb}")
            print(" ")

    # Display tir image and related information
    def display_tir(self):
        if (
            os.path.exists(self.selected_folder_tir)
            and len(self.img_paths_tir) > self.counter
        ):
            self.set_image_tir(self.img_paths_tir[self.counter])  # Display the first image
            self.image_box_tir.setGeometry(
                20 + self.img_panel_width,
                145,
                self.img_panel_width,
                self.img_panel_height,
            )
            self.image_box_tir.setAlignment(Qt.AlignTop)
            # Display the address of the current image, set it to be copyable
            self.tir_name_label.setGeometry(20 + self.img_panel_width, 120, 400, 20)
            self.tir_name_label.setStyleSheet("font-weight:bold;font-size:14px")
            self.tir_name_label.setText("Current SAR: " + self.tir_name[self.counter])
            # Display progress
            self.progress_bar_tir.setGeometry(350 + self.img_panel_width, 120, 100, 20)
            self.progress_bar_tir.setStyleSheet("font-weight:bold;font-size:14px")
            self.progress_bar_tir.setText(f"{self.counter+1} of {self.num_tir}")

    # Display the next image and update the related parameters

    def show_next_image(self):

        # if self.current_rgb_name in self.annotation_dict: #The current image annotation has been submitted
        if self.current_rgb_name in self.attribution_dict:  # The current image annotation has been submitted

            if self.counter < self.num_rgb - 1:  # not the last image
                self.counter += 1
                self.update_current_variable()  # update parameters
                self.display_rgb()
                self.display_tir()
                self.init_pannel()
                # self.display_current_annotation()
                self.display_attribution()
            else:
                self.Warn_Erorr("This is the last image. Thank you for your annotations!")
        else:
            self.Warn_Erorr("Please submit the current annotations first")

    # Display the previous image
    def show_prev_image(self):

        # if self.current_rgb_name in self.annotation_dict:#The current annotation has been submitted
        if self.current_rgb_name in self.attribution_dict:  # The current image annotation has been submitted

            if self.counter > 0:  # not the first image
                self.counter -= 1
                self.update_current_variable()
                self.display_rgb()
                self.display_tir()
                self.init_pannel()
                # self.display_current_annotation()
                self.display_attribution()
            else:
                self.Warn_Erorr("This is the first image, work is just beginning")
        else:
            self.Warn_Erorr("Please submit the current annotations first")

    # Display the image

    def set_image_rgb(self, path_rgb):
        """
        displays the image in GUI
        :param path: relative path to the image that should be show
        """

        pixmap = QPixmap(path_rgb)

        # get original image dimensions
        img_width = pixmap.width()
        img_height = pixmap.height()

        # scale the image properly so it fits into the image window ()
        margin = 20
        if img_width >= img_height:
            pixmap = pixmap.scaledToWidth(self.img_panel_width - margin)

        else:
            pixmap = pixmap.scaledToHeight(self.img_panel_height - margin)

        self.image_box_rgb.setPixmap(pixmap)

    def set_image_tir(self, path_tir):
        """
        displays the image in GUI
        :param path: relative path to the image that should be show
        """

        pixmap = QPixmap(path_tir)

        # get original image dimensions
        img_width = pixmap.width()
        img_height = pixmap.height()

        # scale the image properly so it fits into the image window ()
        margin = 20
        if img_width >= img_height:
            pixmap = pixmap.scaledToWidth(self.img_panel_width - margin)

        else:
            pixmap = pixmap.scaledToHeight(self.img_panel_height - margin)

        self.image_box_tir.setPixmap(pixmap)

    # Add answers to buffer first, then write all at once, can only modify in buffer
    # How to handle modifications: first load image annotations into buffer

    # Add annotated questions to buffer and display annotation information
    def jion_current_dict(self, id, tri_list):
        if self.flag == True:
            if len(tri_list) > 0:
                if len(self.current_qst_dict) == 0:
                    self.current_qst_dict[id] = tri_list
                else:
                    temp_dict = self.current_qst_dict.copy()
                    for k, v in temp_dict.items():
                        if tri_list[0] == v[0]:
                            self.current_qst_dict[k] = tri_list
                        else:
                            self.current_qst_dict[id] = tri_list
                # self.display_current_annotation()
                # self.display_attribution()

    # Submit the file in the buffer

    def submit_annotation(self):

        if len(self.current_qst_dict) > 0 and len(self.current_attri_dict) > 0:
            if self.current_rgb_name == self.current_tir_name:
                # self.annotation_dict[self.current_tir_name] = self.current_qst_dict
                self.attribution_dict[self.current_tir_name] = self.current_attri_dict
                # self.annotation_dict[self.current_tir_name] = self.current_qst_dict
                # self.attribution_dict[self.current_tir_name] = self.current_attri_dict
                self.generate_annotation()
                self.display_anno.clear()
                self.display_anno.append("Annotation submitted and generated successfully")

    # Display error feedback
    def Warn_Erorr(self, str):

        self.display_anno.clear()
        self.display_anno.append(str)

    # Generate annotation file

    def generate_annotation(self):
        save_path_anno = os.path.join(self.save_folder, "annotation_dict.json")
        save_path_attri = os.path.join(self.save_folder, "attribution_dict.json")
        self.attribution_dict["counter"] = self.counter + 1

        # if len(self.annotation_dict) > 0 and len(self.attribution_dict) > 0:
        if len(self.attribution_dict) > 0:
            with open(save_path_anno, "w") as f:
                json.dump(self.annotation_dict, f, indent=4)

            with open(save_path_attri, "w") as f:
                json.dump(self.attribution_dict, f, indent=4)

            self.Warn_Erorr("Annotation file generated successfully")
        else:
            self.Warn_Erorr("File generation failed")

    # Automatically save when the window is closed

    def closeEvent(self, event):
        self.generate_annotation()

    # After switching images, reset the question box
    def init_pannel(self):
        # Remove the image checkbox
        self.init_ans_qst()

    # Display the current annotation content
    def display_current_annotation(self):
        if len(self.current_qst_dict) > 0:
            self.display_anno.clear()
            index = sorted(self.current_qst_dict)
            for q_id in index:
                current_list = self.current_qst_dict[q_id]
                if len(self.current_qst_dict[q_id]) > 0:
                    self.display_anno.append(
                        "Question %s-%s: %s\nAnswer: %s"
                        % (q_id[0], q_id[1:], current_list[0], current_list[1])
                    )
        else:
            self.display_anno.clear()

    # Initialize checkboxes and combo boxes
    def init_ans_qst(self):
        self.flag = False
        for qst in self.question_list:
            if qst.isChecked():
                qst.setChecked(False)
        for index, ans in enumerate(self.ans_list, start=1):
            if index < 22:
                ans.setCurrentIndex(0)
            elif index == 22:
                self.Deduce_Qst_one.setText("Enter your question 1 (one-word answer)")
            elif index == 23:
                self.Deduce_Ans_one.setText("Enter answer")

            elif index == 24:
                self.Deduce_Qst_two.setText("Enter your question 2 (one-word answer)")

            else:
                self.Deduce_Ans_two.setText("Enter answer")
        self.flag = True

    # After switching images, update the related variables
    def update_current_variable(self):

        if len(self.rgb_name) > self.counter and len(self.tir_name) > self.counter:
            self.current_rgb_name = self.rgb_name[self.counter]
            self.current_tir_name = self.tir_name[self.counter]
        if self.current_rgb_name == self.current_tir_name:
            if len(self.annotation_dict) > 0:
                if self.current_rgb_name in self.annotation_dict:
                    self.current_qst_dict = self.annotation_dict[self.current_tir_name]
                else:
                    self.current_qst_dict = {}

                # Update the attribute dictionary
            if len(self.attribution_dict) > 0:
                if self.current_rgb_name in self.attribution_dict:
                    self.current_attri_dict = self.attribution_dict[
                        self.current_tir_name
                    ]
                    if self.contain_dict is not None:
                        if "PresContain" in self.current_attri_dict:
                            self.contain_dict = self.current_attri_dict["PresContain"]
                        else:
                            self.contain_dict = {}
                    if self.traffic_dict is not None:
                        if "Traffic" in self.current_attri_dict:
                            self.traffic_dict = self.current_attri_dict["Traffic"]
                        else:
                            self.traffic_dict = {}
                    if self.residential_dict is not None:
                        if "Residential" in self.current_attri_dict:
                            self.residential_dict = self.current_attri_dict[
                                "Residential"
                            ]
                        else:
                            self.residential_dict = {}
                else:
                    self.current_attri_dict = {}
                    self.contain_dict = {}
                    self.traffic_dict = {}
                    self.residential_dict = {}
                self.display_attribution()
        self.display_anno.clear()
        self.dis_loc_dict = {}

    # Delete some questions, when modifying annotations, just switch to the current image
    def del_annotation(self, q_id="0", q="0"):
        if self.flag == True:
            if q == "0":
                if (
                    len(self.current_qst_dict) > 0
                    and q_id in self.current_qst_dict
                    and len(self.current_qst_dict[q_id]) > 0
                ):
                    self.current_qst_dict.pop(q_id)
            else:
                if len(self.current_qst_dict) > 0:
                    pop_list = []
                    for k in self.current_qst_dict:
                        if k[0] == q:
                            pop_list.append(k)
                    for e in pop_list:
                        self.current_qst_dict.pop(e)

    def ans_mist(self):
        if self.Mist_pick_box.currentIndex() != 0:
            tri_list = []
            qst = "Is there mist in the picture?"
            ans = self.Mist_pick_box.currentText()
            tri_list.append(qst)
            tri_list.append(ans)
            self.jion_current_dict("2", tri_list)
            self.join_attribute_dict("mist", ans)
        else:
            self.del_annotation(q_id="2")
            self.del_attribute_dict("mist")

    def ans_night(self):
        if self.night_pick_box.currentIndex() != 0:
            tri_list = []
            qst = "Is it dark in the picture?"
            ans = self.night_pick_box.currentText()
            tri_list.append(qst)
            tri_list.append(ans)
            self.jion_current_dict("3", tri_list)
            self.join_attribute_dict("night", ans)
            self.display_attribution()  # Add this line to display annotation information
        else:
            self.del_annotation(q_id="3")
            self.del_attribute_dict("night")
            self.display_attribution()  # Add this line to update display


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Annotation_window()
    ex.show()
    sys.exit(app.exec_())
    print("")
