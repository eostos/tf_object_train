#!/bin/bash

# 🧠 TensorFlow Object Detection - Setup Script (Python 3.7 + TF 1.14)

echo "🔄 Updating packages..."
apt-get update

echo "📦 Installing system dependencies..."
apt-get install -y python3.7 python3.7-venv python3.7-dev libgl1-mesa-glx zip protobuf-compiler git wget

echo "🐍 Upgrading pip..."
python3.7 -m pip install --upgrade pip --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

# Install Python packages individually with retries
echo "📦 Installing tensorflow-gpu==1.14"
python3.7 -m pip install tensorflow-gpu==1.14 --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

echo "📦 Installing keras==2.3.1"
python3.7 -m pip install keras==2.3.1 --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

echo "📦 Installing ipykernel"
python3.7 -m pip install ipykernel --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

echo "📦 Installing Pillow==9.2"
python3.7 -m pip install Pillow==9.2 --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

echo "📦 Installing protobuf==3.20.0"
python3.7 -m pip install protobuf==3.20.0 --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

echo "📦 Installing tqdm"
python3.7 -m pip install tqdm --index-url https://pypi.org/simple --timeout 60 --retries 10 -v

echo "🔧 Configuring Git"
git config --global http.postBuffer 524288000

echo "📁 Cloning TensorFlow models repo"
git clone https://github.com/tensorflow/models

echo "📂 Navigating to models/research"
cd models/research

echo "📄 Compiling protobuf files"
protoc object_detection/protos/*.proto --python_out=.

echo "🛠️ Installing TF Object Detection API"
cp object_detection/packages/tf1/setup.py .
python3.7 -m pip install . --index-url https://pypi.org/simple --timeout 60 --retries 10 -v


echo "📁 Creating data directory"
cd ../..
mkdir -p data && cd data

echo "⬇️ Downloading pretrained model"
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

echo "📦 Extracting model"
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

cd ../models/research/object_detection

echo "🚀 Starting training..."
python3.7 model_main.py \
  --pipeline_config_path=/opt/pipeline.config \
  --model_dir=/opt/trainning \
  --num_train_steps=500000 \
  --eval_training_data=True \
  --alsologtostderr
