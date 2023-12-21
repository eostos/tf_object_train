# tf_object_train
Train with tensorflow 1.x , object detection in Docker image.
Python version 3.7
Ubuntu 18.04
tensorflow gpu 1.14-1.15

How to Train ? 
1. Build docker 
bash build.sh
2. Run docker and go inside 
bash run.sh ( remeber change the location your dataset in case the dataset is locally otherwise donwload from cloud in the docker )
sudo docker exec -it  alice_train_tensorflow bash 
3. Download dataset in folder in the container and then configure pipeline with valid and train samples.
4. Generate tfrecords with script.
cd ..
cd models/research/object_detection/
python3.7 model_main.py --pipeline_config_path=../../../pipeline.config --model_dir=../../../trainning --num_train_steps=500000 --eval_training_data=True --alsologtostderr 