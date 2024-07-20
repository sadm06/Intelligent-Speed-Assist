# Intelligent Speed Assist

## Description
Intelligent Speed Assist (ISA) is a project aimed at recognizing speed limits and alerting drivers about potential speed 
limit violations. This project leverages advanced computer vision models, including YOLOv8 for detection and ResNet for 
classification, trained on the German Traffic Sign Detection Benchmark (GTSDB) and the German Traffic Sign Recognition 
Benchmark (GTSRB) datasets, respectively.

## Visuals
![ISA System Output](assets/ISA.gif "ISA System")

## Authors and acknowledgment
This project was implemented by Benjamin Behrouzi and Seyit Duman as part of the final submission for the 
Driver Assistance Systems module.

Special thanks to Phil Harvey for spontaneously providing a new version of Exiftool, which enabled us to extract the 
metadata from our dash cam videos. This was crucial for determining the vehicle speed.

## Usage
1. Ensure you have installed the required packages listed in requirements.txt:
   `pip install -r requirements.txt`

2. Run the script located at `src/system.py` using one of the following commands:
    ```
    python src/system.py <video_path>
    python src/system.py <video_path> <metadata_path>
    ```
    If the metadata path is omitted, the program will search for a metadata file in the same directory as the video 
    file, with the same name. If it does not find the metadata file, no warnings will be shown, but the debug flag will 
    be set to true, allowing you to see the inference process.

    The metadata files should be formatted as plain text files with each line containing a timestamp and a corresponding 
    gps speed value, separated by a comma. The timestamp follows the format YYYY:MM:DD HH:MM:SSZ 
    (with a “Z” indicating UTC time), and the value is a floating-point number. Each line represents a new data entry.
    This can be achieved with [ExifTool](https://exiftool.org) and the following command: 
    `exiftool -ee -p '$gpsdatetime,$gpsspeed' -n <video_path> > <metadata_path>`
    