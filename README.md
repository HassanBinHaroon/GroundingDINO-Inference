# GroundingDINO-Inference
This project represents a GroundingDINO Inference (zero-shot object detection) procedure with both methods (CLI and Script). This implementation will help the reader to know the sequence of commands and exemplifying commands for running a quick zero-shot object detection. Additionally, the reader may get insight into code (script) execution.  This is the Google Colab implementation and has some prerequisites for the complete execution of the script. 

Prerequisite:
The user would need a folder that should be uploaded on the drive. The link to the folder is https://drive.google.com/drive/folders/1ctzsamgcgQ9OpxclnNlj7iLYsbbbh2R3?usp=share_link. The user needs to download that folder and upload it to the drive.

1. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jm2YiPdutOg68MVhDDlg8xt99PK7Piic)

2. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zIKmMZTY8Kcy0xIltKx0lg1ccEFkiF0g)

3. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kX1IeVrwA9J97lN1MqsFF_afambuinv8#scrollTo=4dEL6xotpB_A)

4. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJye4465EpEZ_h1loUcyPsPecKhhT9ka#scrollTo=KxbH9rKcHdjn)

## Table Of Contents

  1. Demo Results
  2. Inference Steps (CLI):
  3. Inference (Code):

## Demo Results

### Demo1

![](https://github.com/HassanBinHaroon/GroundingDINO-Inference/blob/main/Demo%20Images/img1.jpg)

### Demo2

![](https://github.com/HassanBinHaroon/GroundingDINO-Inference/blob/main/Demo%20Images/img2.jpg)

### Demo3

![](https://github.com/HassanBinHaroon/GroundingDINO-Inference/blob/main/Demo%20Images/annotated_image.jpg)

### Demo4

![](https://github.com/HassanBinHaroon/GroundingDINO-Inference/blob/main/Demo%20Images/annotated_image(1).jpg)

### Demo5

![](https://github.com/HassanBinHaroon/GroundingDINO-Inference/blob/main/Demo%20Images/annotated_image(2).jpg)

## Inference Steps (CLI): [GroundingDINO_Trial1.ipynb]

### Step1

    from google.colab import drive
    drive.mount('/content/drive')

### Step2

    cd drive/MyDrive/GroundingDINO/

### Step3

    pip install -q -e .
    
### Step4
 
    python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -i frame1.jpg -o "output" -t "bike . person . car ." 

## Results Visualization (CLI)

Go to the output folder that would be created automatically.

## Inference (Code): [GroundingDINO_Trial2.ipynb]

    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    IMAGE_PATH = "im1.jpg"
    TEXT_PROMPT = "persons . sofas . fans"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
           model=model,
           image=image,
           caption=TEXT_PROMPT,
           box_threshold=BOX_TRESHOLD,
           text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)


## Results Visualization (Code)

An annotated image will be created in the code directory.

## References

https://github.com/IDEA-Research/GroundingDINO

## Important Announcement

This is just the beginning of the project. More options and features may be provided in the future.
