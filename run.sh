arg_tag=alice_train_tensorflow:latest
arg_name=alice_train_tensorflow
#debug
docker_args="--entrypoint /bin/bash --name $arg_name --restart unless-stopped --gpus all -v /edgar1/datase_license_plate:/opt/  --log-driver local --log-opt max-size=10m --net host -dt $arg_tag"

#prod
#docker_args="--name $arg_name --restart unless-stopped -v /edgar1/datase_license_plate:/opt/  --log-driver local --log-opt max-size=10m --net host -dt $arg_tag  bash"
echo "Launching container:"
echo "> docker run $docker_args"
docker run $docker_args
