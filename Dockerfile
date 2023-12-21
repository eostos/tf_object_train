# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#ARG CUDA_VERSION=10.0
#ARG OS_VERSION=18.04

#FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
#LABEL maintainer="NVIDIA CORPORATION"
#ARG CUDA_VERSION=10.0
#ARG OS_VERSION=18.04

#FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu${OS_VERSION}
#LABEL maintainer="NVIDIA CORPORATION"
#FROM ubuntu:20.04
FROM gouchicao/mmdetection:latest
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.0-base
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update 
RUN apt-get update --fix-missing



ENTRYPOINT ["sh","start.sh"]
