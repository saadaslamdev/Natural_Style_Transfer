# Neural-Style-Transfer (NST)

Neural Style Transfer is the ability to create a new image (known as a pastiche) based on two input images: one representing the content and the other representing the artistic style.

## Overview <a name="overview"></a>

Neural style transfer is a technique that is used to take two images—a content image and a style reference image—and blend them together so that output image looks like the content image, but “painted” in the style of the style reference image.

## Getting Started <a name="getting-started"></a>

### File Description <a name="description"></a>

    Neural-Style-Transfer
        ├── data
        |   ├── content-images (Target images)
        |   |── output-images (Result images)
        |   ├── style-images (Style images)
        ├── models/definitions     
        │   ├── vgg19.py   <-- VGG19 model definition
        ├── app.py  <-- streamlit app to run in web
        ├── NST.py  <-- the main python file to run in CLI
        └── README.md

### Dependencies <a name="dependencies"></a>
*    Python 3.9+
*    Framework: PyTorch
*    Libraries: os, numpy, cv2, matplotlib, torchvision

### Usage <a name="usage"></a>

```
    $ pip install -r requirements.txt
```

#### To implement Neural Style Transfer on images of your own:

1. Clone the repository and move to the downloaded folder:
```
    $  git clone https://github.com/nazianafis/Neural-Style-Transfer
```

2. Move your content/style image(s) to their respective folders inside the `data` folder.

3. Go to `NST.py`, and in it, set the `PATH` variable to your downloaded folder. Also set `CONTENT_IMAGE`, `STYLE_IMAGE` variables as your desired images:
```
    $ PATH = <your_path>
   
    $ CONTENT_IMAGE = <your_content_image_name>
    $ STYLE_IMAGE = <you_style_image_name>
```
4. Run `NST.py`:
```
    $ python NST.py
```
5. Find your generated image in the `output-images` folder inside `data`.

## Web APP View

Following is the web app view of the project where you can upload your content and style images, edit the weights and iterations, and get the result.

Run `app.py`:
```
    $ streamlit run app.py
```

<img src="" width="700"/>
