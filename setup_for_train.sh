#https://www.tensorflow.org/install/source?hl=es-419#linux
#https://universe.roboflow.com/bright-line-solutions/license-plates-detection-anpr/dataset/3
#https://universe.roboflow.com/plat-kendaraan/vehicle-and-license-plate/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
#https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
apt-get update
apt-get install -y python3.7 python3.7-venv python3.7-dev libgl1-mesa-glx  zip protobuf-compiler
python3.7 -m pip install --upgrade pip
python3.7 -m pip install tensorflow-gpu==1.14
python3.7 -m pip install keras==2.3.1
python3.7 -m pip install ipykernel
python3.7 -m pip install Pillow==9.2
python3.7 -m pip install protobuf==3.20.0

#Pillow==6.1.0

apt-get install git
pip install tqdm 
git config --global http.postBuffer 524288000
git clone  https://github.com/tensorflow/models

#navigate to /models/research folder to compile protos
ls
cd models/research

# Compile protos.

protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py . 

python3.7 -m pip install .
cd ../..
mkdir data
cd data
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ..
cd models/research/object_detection/
python3.7 model_main.py --pipeline_config_path=/opt/pipeline.config --model_dir=/opt/trainning --num_train_steps=500000 --eval_training_data=True --alsologtostderr