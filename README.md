# Cancer-Brain-3.0

The main purpose of this repository is to provide AI tools for supporting brain cancer detection and classification.

## Features

- **Training**: Train brain cancer classification models (LGG/HGG)
- **Inference**: Run inference on brain scan images using Triton Inference Server
- **Web Application**: Interactive drag-and-drop web interface for image classification

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd Cancer-Brain-3.0
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Training

Train a brain cancer classification model:

```bash
python train.py --project-name <name> --dataset-dir <path> --model <model_name> --batch <batch_size> --lr <learning_rate> --epochs <num_epochs> --dataset-ratio <ratio>
```

### Training Arguments

- `project-name`: Name of the training project
- `dataset-dir`: Directory containing the dataset
- `model`: Model architecture to use
- `batch`: Batch size
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `dataset-ratio`: Dataset split ratio

## Inference

### Command Line Interface

Run inference on a single image:

```bash
python infer.py --orin_ip <ORIN_IP_ADDRESS> --image_path <path_to_image>
```

Arguments:
- `--orin_ip`: IP address of the ORIN device running Triton Inference Server (required)
- `--image_path`: Path to the input image (default: "test.jpg")

### Web Application

Launch the interactive web application with drag-and-drop functionality:

```bash
export ORIN_IP=<ORIN_IP_ADDRESS>
python app.py
```

Or run in one line:
```bash
ORIN_IP=<ORIN_IP_ADDRESS> python app.py
```

The web app will be available at `http://localhost:5000` (or the port specified by the `PORT` environment variable).

#### Web App Features

- **Drag & Drop**: Simply drag and drop an image file onto the upload area
- **Click to Browse**: Click the upload area to browse and select an image
- **Image Preview**: Preview the uploaded image before submission
- **Real-time Results**: Get instant classification results with:
  - Predicted class (LGG or HGG)
  - Confidence score
  - Raw logits values

Supported image formats: PNG, JPG, JPEG, GIF, BMP

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Flask (for web application)
- Triton Inference Server client
- See `requirements.txt` for complete list

## Project Structure

```
Cancer-Brain-3.0/
├── app.py                 # Flask web application
├── infer.py              # Inference script
├── train.py              # Training script
├── templates/
│   └── index.html        # Web app frontend
├── requirements.txt      # Python dependencies
└── README.md            # This file
```
