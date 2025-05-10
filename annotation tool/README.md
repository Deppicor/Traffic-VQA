# Annotation software instructions

## I. Software introduction

This tool is a graphical annotation software developed based on PyQt5, which supports problem annotation for aligned optical images (OPT) and thermal imaging images (THE). Users can selectively annotate image matching, weather information, scene type, traffic cognition, etc. through interface operations, and automatically save them to the specified directory. It is suitable for building datasets for tasks such as MultiModal Machine Learning visual question answering and image understanding.

## II. Environmental dependence

Python ≥ 3.6

Required libraries to be installed:

`pip install numpy opencv-python PyQt5`

## III. Start-up method

Make sure you have placed the `annotation_tool.py `in the root directory of your project, and then run:

bash

`python annotation_tool.py`

## IV. Instructions for use

### 1. Interface layout

The overall layout of the interface is as follows: there are three buttons in the upper left corner, namely "Browse OPT", "Browse THE", and "Browse SAVE", which are used to select the OPT image folder, THE image folder, and the save path of the annotation file in order. The middle left side of the interface is the image display area. After the user selects an image, the corresponding optical image and thermal image will be displayed side by side in this area, and the current image name and annotation progress information will be displayed above the image. The lower left side of the interface is the annotation content display area. The annotation information that the user has confirmed to save will be displayed here in order for easy review and management. The right side of the interface is the annotation option area, which lists all optional question types (Match, Theme, Distance/Location, Contain/Presence, Deduce, Traffic, Residential, Agricultural, Industrial, UAV). Users can check any question type and select its corresponding subclass option below for detailed annotation.

### 2. Select a folder

Before using this annotation tool, users need to first complete the selection of three key folders. Click the "Browse OPT" button in the upper left corner of the interface to select the folder for storing optical images; click the "Browse THE" button to select the folder for storing thermal images; click the "Browse SAVE" button to select the folder for saving annotation results. After the three path selections are completed, the system will automatically load and align each pair of optical and thermal images, and display the current image name and annotation progress information above the image to prepare for subsequent image annotation operations.

### 3. Annotate image-level attributes

When labeling image-level attributes, users need to describe their macro features based on the overall information of the entire image.

After checking "Match", users can select the following subclass attributes in order: whether the image matches (almost match, partial match, not match), and weather simulation information, including fog conditions (mist, not mist, not sure) and low light conditions (dark, not dark, not sure).

After selecting "Theme", users need to determine whether the image belongs to a residential area (Residential or n-Residential), and select the macro category of the scene (Urban or Rural) to indicate the theme environment type of the image.

### 4. Mark LocDis

When labeling "Distance/Location" (abbreviated as LocDis) problems, users need to provide a detailed description of the positional relationship between two objects (object A and object B) in the image.

First, the user needs to select the location of object A, specify whether it is an area (with options of none or cluster), and then select the category of object A from the drop-down list (including woodland, grassland, other vegetation area, pond, river, ditch, sea, lake, other water area, wide road, narrow road, low-rise residential building, high-rise residential building, low-rise non-residential building, high-rise non-residential building, agricultural area, wasteland, intersection, parking area, park, concrete floor, basketball court, football field, baseball field, athletic track, tennis courts, pier, beach, airport, apron).&#x20;

Next, the user needs to select the position of object B, whether it is a region (none, cluster), and the category of object B in the same way. Then, the distance relationship between AB needs to be specified, as well as the orientation of B relative to A (above, below, left, right, upper left, upper right, bottom left, bottom right).

After completing the option filling, click the Confirm button to save the annotation information of the relative spatial relationship between the A-B objects.

In addition, by clicking the "Length and Distance" button, an optical image will pop up on the interface. Users can select the closest distance between two objects to be measured by clicking on two points. The system will automatically calculate the actual length and display it.

After completing the annotation of the relationships between all objects, click the "Input End" button, and the system will automatically summarize all the currently saved LocDis information.

### 5. PresContain

When annotating PresContain-type questions, users need to check "Contain/Presence" and then complete the annotation task according to the following process.

First, the user needs to select a LandCover Class, with options including: agricultural area, airport, apron, building, beach, pier, intersection, park, parking area, sports field, road, concrete floor, vegetation area, wasteland, water area.

After selecting a LandCover Class, further select its corresponding Subset Class.

If it is an agricultural area, the corresponding Subset Class is: agricultural area.

If it is airport or apron, the corresponding Subset Class is: apron.

If it is a building, the optional Subset Class is: low-rise residential building, high-rise residential building, low-rise non-residential building, high-rise non-residential building.

If it is beach, pier, intersection, park, parking area, concrete floor, wasteland, then their respective Subset Classes have the same name.

If it is a sports field, the optional Subset Classes are: basketball court, baseball field, football field, tennis courts, athletic track.

If it is a road, the optional Subset Class is: wide road, narrow road.

If it is a vegetation area, the optional Subset Class is: woodland, grassland, other vegetation areas.

If it is a water area, the optional Subset Class is: ditch, pond, river, sea, lake, other water areas.

After completing the category selection, the user needs to further select the specific attributes of the object, including:

Number, Location, Shape (optionally none, Straight, Curved, Triangle, Square, Rectangle, other quadrilater, Rotundity, other shape), Area, Length, Distribution (optionally none, Clustered Distribution, Isolated Distribution, Dense Distribution, Random distribution, Uniform distribution), Better (image quality judgment, choose optical, thermal, or almost same)

When it is necessary to measure the area of the object, click the "Measure Area" button, and the system will pop up the optical image interface. Users can circle the target area clockwise to automatically obtain the area data.

After completing the attribute filling, click the "Confirm" button, and the annotation information of the current object will be saved as a piece of data. Users can continue to select other objects and repeat the operation. Each confirmation will generate a new annotation record.

After completing all object annotations, click the "Input End" button, and the system will automatically summarize all saved data.

To modify, users can select a marked object and click the "Delete Current LandCover" button. The object and all its attribute information will be deleted together.

### 6. Deduce labeling

When performing Deduce annotation, users need to first check the "Deduce" option. Then, in the four text boxes provided by the interface, enter two sets of questions and answers in order: enter question one in the first text box "Please enter your question one", enter the answer to the question (the answer must be one word) in the second text box "Please enter the answer", and then click the "submit first" button to submit; then enter question two in the third text box "Please enter your question two", enter the answer to the question (also one word) in the fourth text box "Please enter the answer", and click the "submit second" button to submit. After completing the above operations, the two sets of questions and their corresponding answers will be successfully saved as reasoning data.

### 7. Traffic labeling

When labeling Traffic-related issues, users need to select "Traffic" and then complete the labeling task according to the following process.

Firstly, the user needs to select a LandCover Class. The options include: road type, vehicle, pedestrian, road facility, road condition, vehicle traffic violation, non-motor vehicle violation, pedestrian traffic violation, vehicle behavior, non-motor vehicle behavior, pedestrian behavior, abnormal traffic situation, and traffic participant interaction.

After selecting a LandCover Class, further select its corresponding Subset Class.

If it is a road type, the optional Subset Classes are: main city road, street, quick road, residential street, alley, intersection, lane merge, pedestrian crossing, bridge, non-motorized road, unpaved road, bus lane, gridline, overhead walkway, other paved road.

If it is a vehicle, the optional Subset Class is: car, large vehicle, other vehicle.

If it is pedestrian, the optional Subset Class is: single pedestrian, pedestrian group.

If it is a road facility, the optional Subset Classes are: motor vehicle parking spot, non-motorised parking spot, road divider.

If it is a road condition, the optional Subset Classes are: normal pavement, damaged pavement, road construction.

If it is a vehicle traffic violation, the optional Subset Classes are: illegal parking, go against one-way traffic, illegal lane change, run the red light, vehicle on solid line.

If it is a non-motor vehicle violation, the optional Subset Classes are: illegal passenger carrying, wrong-way driving, running red light, improper lane usage, improper parking, no safety helmet.

If it is a pedestrian traffic violation, the optional Subset Class is: failure to use crosswalks, walking on non-sidewalks, running the red light, other violations.

If it is vehicle behavior, the optional Subset Classes are: lane change, vehicle turn, vehicle U-turn, overtake, vehicle queuing, traffic congestion, too close to another car.

If it is non-motor vehicle behavior, the optional Subset Class is: waiting at traffic light, normal driving in non-motor lane.

If it is pedestrian behavior, the optional Subset Classes are: wait for a traffic light, walk on the crosswalk, walk on the sidewalk.

If it is an abnormal traffic situation, the optional Subset Class is: traffic accident, traffic jam;

If it is a traffic participant interaction, the optional Subset Classes are: vehicle yielding to pedestrian, vehicle waiting for boarding, vehicle waiting for alighting, bus temporary stop, vehicle entering parking lot, vehicle exiting parking lot.

After completing the category selection, the user needs to further fill in the specific attributes of the object, including:

Number (quantity), Location (location), Quality (image quality judgment, choose optical, thermal or almost the same).

After completing the attribute filling, click the "Confirm" button, and the annotation information of the current object will be saved as a piece of data. Users can continue to select other objects and repeat the operation. Each confirmation will generate a new annotation record.

After completing all object annotations related to Traffic, click the "Input End" button, and the system will automatically summarize all saved data and write it into the annotation file.

To modify, users can select a marked object and click the "Delete Current Traffic LandCover" button. The object and all its attribute information will be deleted together.

### 8. Mark **Residential**

When annotating Residential questions, users need to check "Residential" and then complete the annotation task according to the following process.

First, the user needs to select a LandCover Class, with options including: living environment, construction type.

After selecting a LandCover Class, further select its corresponding Object.

If it is a living environment, Object options include: recreational area, commercial area, construction area, river, lake, linear walkway, curved walkway, no visible walkway.

If it is a construction type, the optional options for Object include: low-rise residential buildings, high-rise residential buildings, low-rise non-residential buildings, and high-rise non-residential buildings.
Then, the user needs to further select the properties of the object, including:
Number, Location, Quality (optical, thermal, almost the same).

After the selection is done, click the "Confirm" button, and the current annotated data will be saved as a record. Users can continue to select other Objects and click Confirm to save new annotated data.
After all the labeled objects related to Residential in the image have been labeled, click the "Input End" button, and the system will uniformly write all saved data into the annotation file.

If the user selects a marked Object and clicks the "Delete Current Residential LandCover" button, the object and all corresponding annotation information will be deleted.

### 9. Agricultural labeling

After checking "Agricultural", users can select in order: whether there are agricultural roads (yes, no), whether there are agricultural water bodies (yes, no).

### 10. Labeling Industria

After checking "Industrial", users can select in order: whether there are industrial facilities (yes, no), industrial scale (small, medium, large), industrial location (top, bottom, left, right, center, left top, top right, bottom left, bottom right).

### 11. Mark UAV

After selecting "UAV", users can select in order: drone shooting height (150-250, 250-400, 400-550, none), drone shooting angle (vertical, oblique)

### 12. Submit

Click "Pre" to return to the previous set of pictures.

Click "Next" to jump to the next set of pictures.

Click "Submit" to submit all the content annotated in the current image.

Click "Generate" to generate the annotation file.

## V. Notes

Please ensure that OPT matches the naming of THE images one-to-one.

You can choose to skip some questions or not check them all.

The annotation file will be automatically named after the image name.

## VI. Contact and feedback

If you have any suggestions, feature improvements, or bug feedback, please feel free to contact the developer team via email.

