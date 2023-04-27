# GroundingDINO-Inference
This project represents a GroundingDINO Inference (zero-shot object detection) procedure. This implementation will help the reader to know the sequence of commands and exemplifying commands for running a quick zero-shot object detection. This is the Google Colab implementation and has some prerequisites for the complete execution of the script. 

Prerequisite:
The user would need a folder that should be uploaded on the drive. The link to the folder is https://drive.google.com/drive/folders/1ctzsamgcgQ9OpxclnNlj7iLYsbbbh2R3?usp=share_link. The user needs to download that folder and upload it to the drive.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kX1IeVrwA9J97lN1MqsFF_afambuinv8#scrollTo=4dEL6xotpB_A)

## Inference Steps:

### Step1

    from google.colab import drive
    drive.mount('/content/drive')

### Step2

    cd drive/MyDrive/GroundingDINO/

### Step3

    pip install -q -e .
    
### Step4
 
    python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -i frame1.jpg -o "output" -t "bike . person . car ." 

## Results Visualization

Go to the output folder that would be created automatically.
   
