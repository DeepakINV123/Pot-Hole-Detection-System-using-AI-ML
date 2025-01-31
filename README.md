# Pot-Hole-Detection-System-using-AI-ML
For this project, I am training a YOLO11s object detection model to detect potholes using Google Colab. The dataset consists of 500 images with varying backgrounds and lighting conditions to enhance model robustness.

Online sources for dataset: roboflow universe, Kaggle, google open images v7 dataset.
*Have photos with different background and light adjustments, and have a minimum of 100 to 150 photos for training.

Online tools to label photos: Label Studio.
* You can add all the images and then label them separately. Before downloading Label studio you need to download Anaconda for managing Python environment.(anaconda.com/download/success).
You can use default steps for installation. Once anaconda is downloaded, you will find Anaconda prompt, you can open it and you'll see that a command prompt opens. In the command prompt we will create a new python environment. "conda create --name yolo-env1 python=3.12"  
After this it will ask for [y/n], give 'y' for yes.
Once the environment is created we have to activate it by typing "conda activate yolo-env1"
You will see that you are no longer in the base workspace, you would have shifted to the (yolo-env1) workspace. It will take a few minutes to install.

Then type "pip install label-studio".
This command might now work so try taking AI help. I tried using 
"pip install label-studio -i https://mirrors.aliyun.com/pypi/simple/"

Then to start label studio you have to type "label-studio start"
After this the label studio will start in a new web browser. You can either login or signup. Then click on create project.


* In label studio create a new project, you'll see 3 tabs on top (Project name, data import and Labeling setup), give a new project name and then go to the data import
and import the data of all the photos using drag and drop format. Drag and drop 100 images at a time, don't exceed this limit. Now in the Labeling setup select the object detection with bounding boxes. Remove all the predefined labels given for example airplane and car. Give your own labels line wise (These labels are basically names that you want to see when you scan the object and give underscore instead of space in label). Before saving you'll see that each label has its own color. Then save the project. 

*Click the first picture in the labeling interface and then when the picture opens select the labels you want to choose and drag a box around it and then click submit.
Once labeling is done for all the images click on the project name on top and then click export and select Yolo format and export.
Label studio will automatically package data into a zip file. You can rename the zip file (data.zip) and save it in the same folder as the images.
This will be used to train the Yolo model afterwards.
The data.zip contains (images folder, labels folder, classes.txt, notes.json)

Go to your own google collab notebook and then make sure to login first. The above link is to other colab notebook for reference.

First go to runtime and select language as "python3" and select a "T4 GPU" as hardware accelerator then save it and click 'connect' on the top right tab. 
When you scroll down you'll see a verification for gpu. You can type this below given shell script in your personal collab notebook.
The command is "!nvidia-smi". It is used to check the status of the allocated GPU.


 * Next you need to upload your zip file (data.zip) in the colab instance. You will see a folder logo at the last of the left pane and then you can drag and drop your data.zip. (Sample data and other reference folders will appear automatically).


* Once the .zip file is present you need to unzip it to create folders to hold your data. Use the given command. 
"!unzip -q /content/data.zip -d /content/custom_data". Here this is a shell script it takes the data from '/content/data.zip' and then unzips it and stores the extracted data into '/content/custom_data'.

*Now we have to divide the data into 2 folders one is training and the other is validation. The training data will be used to train the model here all the images will be given to the model. The validation folder will be used to run checks whether the model is trained in the right manner or not. 
There is a python script that automatically divides the data into two folders it adds 90% of the data into the training folder and 10% into the validation folder. (train_val_split.py)

Load the 'train_val_split.py' to the workspace as well and run the below command.
!python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9"


Here the first line is a shell script that takes the file from the web link and then saves it int he colab notebook at '/content/train_val_split.py' location. The second line executes the file in python environment. The file is run and the dataset to be split is (--datapath="/content/custom_data").
And  (--train_pct=0.9") specifies that 90% of file should be used for training.



Remember that this step is not practically tested.
Once this step is done a new file will be created in collab named as 'Data' it would contain two folders named 'train','validation'

 
*Once the dataset is divided we are ready to go forward with regards to functioning.
To train the YOLO model we need to download a library named as 'ultralytics' to provide a comprehensive framework for training YOLO models, particularly the newer versions like YOLOv5, YOLOv8, and beyond.  To download type shell script
"!pip install ultralytics"


* Before we start the training we need to create a training configuration file. A training configuration file is a file that contains settings and parameters that would be used to train the machine learning model. It mostly contains training and validation datasets, paths of other data if any, information about architecture of the model, other parameters like learning rate (it means how fast the weight are adjusted i the neural network), number of epochs (it means how many times the model is passed through the entire dataset), batch size (It means number of examples used to train), name of classes(its nothing but the labels), number of classes.

Here we create a Ultralytics training configuration YAML file.
Use the below code to create the "data.yaml" file.  Type this code in colab notebook and run it.

"import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):

  # Read class.txt to get class names
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Create data dictionary
  data = {
      'path': '/content/data',
      'train': 'train/images',
      'val': 'validation/images',
      'nc': number_of_classes,
      'names': classes
  }

  # Write data to YAML file
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Created config file at {path_to_data_yaml}')

  return

# Define path to classes.txt and run function
path_to_classes_txt = '/content/custom_data/classes.txt'
path_to_data_yaml = '/content/data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

print('\nFile contents:\n')
!cat /content/data.yaml"

---------------------------------------------------------------------------------------------

Now a "data.yaml" file will be created in the Files area of colab notebook.


*Now we can run training but first its better to select which YOLO model should be selected.

For image detection of seen items like automobiles, number plate or in terms to know accuracy its better to use YOLO11s.
For training we need to set parameters like number of epoch. For a model with 200images a epoch of 60 is considered as the model will be trained for 60 times on the same dataset. You can take an approximation based on your dataset.

You also need to select the resolution, if you want your model to work fast then use lower resolution like 480x480 or its better to go with default resolution of 640x640.



Use this command to run training:
"!yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640"

You can directly run this command, there is no need to download anything else as everything is already mentioned in the ultralytics library.

After running the training data it passes the prints the accuracy of the validation data for each epoch.
Once the training is done you'll see "yolo11n.pt","yolo11s.pt" and "runs". The best model weights will be stored in the 'runs' file--> detect--> train-->weights-->best.pt


Sometimes you run the model for different epochs and resolution that time multiple detects gets formed which makes it difficult to understand and find the results, so we need to delete the 'run' directory, but this directory will not get deleted. So to delete it use the following command:

"import shutil
dir_path = 'path_of_the_folder_in_colab'
shutil.rmtree(dir_path)" 


And also remember to delete the yolo11n.pt and yolo11s.pt from files instance.


An sometimes when you want to download a file from google colab you cannot download it directly so you need to zip it first. So for that use this command:
!zip -r "name.zip" "path_of_file_to_be_zipped"
------------------------------------------------------------------------------------------------------
*After the model training is completed we start the model testing.

Run this command:
"!yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True"
This command tell YOLO to run prediction for weights stored in runs with the validation images and save the data. The saved data consists of prediction results and images in the output directory.


"import glob
from IPython.display import Image, display
for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg')[:10]:
  display(Image(filename=image_path, height=400))
  print('\n')"
  


This command finds the path names and displays the images in the notebook. It displays the first 10 images from the 'content/run/detect/predict/directory' (images that have already been predicted).


*Now to deploy the model.
First we will zip and download the trained model into a folder named (my_model). It consist the best weights renamed to 'my_model.pt' and also contains the training results.

Run the below code for the above task:
"# Create "my_model" folder to store model weights and train results
!mkdir /content/my_model
!cp /content/runs/detect/train/weights/best.pt /content/my_model/my_model.pt
!cp -r /content/runs/detect/train /content/my_model

# Zip into "my_model.zip"
%cd my_model
!zip /content/my_model.zip my_model.pt
!zip -r /content/my_model.zip train
%cd /content" 


Now you can download the "my_model.zip" from the files bar. Store the .zip file in the same folder that contains data.zip and images and then extract it there.

Now go to the "anaconda prompt", if it is still running then label studio press ctrl+C to stop it or else you can reopen it again and activate the environment by typing 
"conda activate yolo-env1"


Now copy my_model path and type.
(yolo-env1)> "cd C:\Users\taxig\Documents\yolo\my_model"
(yolo-env1) C:\Users\taxig\Documents\yolo\my_model> "pip install ultralytics"

The above step would take some time and it would also install import libraries like OpenCV-Python, Numpy, and PyTorch).

Incase you want the GPU enabled version of Pytorch you would have to enable another command:
(yolo-env1) C:\Users\taxig\Documents\yolo\my_model> pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


--> Next you would have to download the yolo_detect.py script into the my_model folder using:

(yolo-env1) C:\Users\taxig\Documents\yolo\my_model> curl -o yolo_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo_detect.py

(yolo-env1) C:\Users\taxig\Documents\yolo\my_model> python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720

The above line will run the script live using USB camera and show the video in the given resolution. You can press 'Q' to exit the program.

If you want to run the script on a pre-recorded video then add the video into my_model folder and then type.
(yolo-env1) C:\Users\taxig\Documents\yolo\my_model> python yolo_detect.py --model my_model.pt --source video_name.mp4


The yolo_detect.py file is given in the repository.
