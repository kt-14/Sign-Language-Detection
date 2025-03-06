# Sign Language Detection Model

## Description
This project is a sign language detection model that utilizes machine learning to recognize hand gestures. The model employs a trained Random Forest classifier to provide real-time predictions based on gestures captured from a webcam.

### The project includes:
- **train.py**: A script to train the sign language detection model.
- **implementModel.py**: A script to implement the trained model and provide real-time predictions.
- **generate_spec.py**: A script to generate a `.spec` file tailored to the user's environment for building an executable.
- **make_exe.py**: A script to create an executable file from the generated `.spec` file using PyInstaller.

## Prerequisites
Ensure you have all required dependencies installed by running:
```sh
pip install -r requirements.txt
```

## Installation
Clone the repository:
```sh
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
```
Ensure you have the necessary data files in the project directory if required for training.

## Usage
### Train the Model
Run the `train.py` script to train the model and save it for later use:
```sh
python train.py
```
This script will generate and save the trained model file.

### Implement the Model
To use the trained model for sign language detection, run:
```sh
python implementModel.py
```
This will open your webcam and display the predictions for detected hand gestures in real-time.

### Generate the Spec File
To generate a `.spec` file for creating an executable, run:
```sh
python generate_spec.py
```
This will create a file named `implementModel.spec` in the project directory.

### Create the Executable
After generating the `.spec` file, build the executable by running:
```sh
python make_exe.py
```
This will create an executable file in the `dist/implementModel` directory.

### Finding the Executable
Once the executable has been created, navigate to the following directory to find it:
```sh
dist/implementModel/
```
Inside this folder, you will find the `implementModel.exe` file, which you can run to use the sign language detection model without needing a Python environment.
