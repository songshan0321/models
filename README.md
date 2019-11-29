# MobileNet-SSD with Tensorflow, ROS & NCS2

This repository provides a platform to easily **train your cutomized SSD-Mobilenet model with Tensorflow** for object detection and **inference on Intel Neural Compute Stick 2 (NCS2) with ROS**. It is forked from [tensorflow/models](https://github.com/tensorflow/models.git).

**MOST IMPORTANTLY**, a super detailed [step-by-step instruction](#Step-by-step Instruction) is included! 

## Pre-requisites

- Ubuntu 16.04
- ROS-Kinetic

- CUDA10.0 (*Optional for faster training)
- OpenVINO R3 (To run on NCS2 with optimized model)
- Intel Realsense SDK 2.0 (*Optional for inference on NCS2)

## Directory Structure

Before proceeding further, I want to discuss directory structure that I will use throughout the tutorial.

```
models/research/
    ├── data
    |   ├── Annotations - customized dataset annotations
    |   ├── ImageSets/Main - train test split information
    |   └── JPEGImages - customized dataset images
    ├── model_dir
    |   ├── train - save trained model
    |   └── eval - save results of evaluation on trained model
    ├── tf_records - 
    ...
```

## Step-by-step Instruction

1. ### Environment Setup

   1. Install Tensorflow.

      ```bash
      pip install tensorflow-gpu==1.14
      # or
      pip install tensorflow-gpu
      ```

   2. Clone this repository into your home directory.

      ```bash
      cd && git clone https://github.com/songshan0321/ros_ssd_tensorflow.git
      mv ros_ssd_tensorflow models
      ```

   3. Setup Tensorflow training environment.

      ```bash
      cd ~/models/research
      
      # Compile Protobuf
      protoc object_detection/protos/*.proto --python_out=.
      
      # Add Python path (Need to do this everytime in a new terminal)
      export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
      
      # Test Object Detection API is working properly
      python object_detection/builders/model_builder_test.py
      
      # Congratz, environment setup is done! 
      ```

2. ### Data Preparation

   1. Create a data folder in PASCAL VOC format.

      ```bash
      # data/
      # ├── Annotations/
      # ├── JPEGImages/
      # └── ImageSets/Main/
      
      cd ~/models/research/
      mkdir -p data/Annotations data/JPEGImages data/ImageSets/Main
      ```

   2. Collect images and put them into `models/research/data/JPEGImages`

   3. Rename images and annotations sequentially in your data directory.

      ```bash
      # Rename images
      cd ~/models/research/data/JPEGImages/
      a=0
      for i in *.jpg; do
        new=$(printf "%05d.jpg" "$a")
        mv -i -- "$i" "$new"
        let a=a+1
      done
      ```

   4. Annotate your images in PASCAL VOC format using [labelImg](https://github.com/tzutalin/labelImg).

      ```bash
      git clone https://github.com/tzutalin/labelImg.git
      sudo apt-get install pyqt5-dev-tools
      sudo pip3 install -r requirements/requirements-linux-python3.txt
      make qt5py3
      python3 labelImg.py
      ######################################################
      # Set 'image dir' as ~/models/research/data/JPEGImages
      # Set 'save dir' as ~/models/research/data/Annotations
      # Start labeling! 
      ######################################################
      ```

   5. After finish annotations, split dataset into trainval and test. You can change the parameters in the script, e.g. trainval_ratio, default = 0.9. To reproduce the same split, set the seed_int = 1.

      ```bash
      python ~/models/research/trainval_test_split.py
      ```

   6. Generate TF Record for your dataset.

      ```bash
      cd ~/models/research
      # Create train records
      python object_detection/dataset_tools/create_my_tf_record.py \
          --label_map_path=data/label_map.pbtxt \
          --data_dir=data --set=trainval \
          --output_path=tf_records/trainval.record
      # Create test records
      python object_detection/dataset_tools/create_my_tf_record.py \
          --label_map_path=data/label_map.pbtxt \
          --data_dir=data --set=test \
          --output_path=tf_records/test.record
      ```

      

3. ### Model Training

   1. Train a Tensorflow model, you can change the hyperparameters in [ssd_mobilenet_v2_coco.config](ssd_mobilenet_v2_coco.config).

      ```bash
      cd ~/models/research
      mkdir -p model_dir/train model_dir/eval
      python object_detection/legacy/train.py --logtostderr  --train_dir=model_dir/train --pipeline_config_path=ssd_mobilenet_v2_coco.config
      
      # Monitoring Progress with Tensorboard
      tensorboard --logdir=model_dir/train
      ```

4. ### Export Inference Graph

   1. Export an inference graph based on the selected checkpoint file.

      ```bash
      python object_detection/export_inference_graph.py \
          --input_type image_tensor \
          --pipeline_config_path ssd_mobilenet_v2_coco.config \
          --trained_checkpoint_prefix model_dir/train/model.ckpt-[ITER-NUM] \
          --output_directory exported_graphs/0
      ```

5. ### Model Evaluation

   1. Test model accuracy

      ```bash
      python object_detection/legacy/eval.py \
          --logtostderr \
          --pipeline_config_path=ssd_mobilenet_v2_coco.config \
          --checkpoint_dir=model_dir/train/ \
          --eval_dir=model_dir/eval/
          
      # Monitoring Progress with Tensorboard
      tensorboard --logdir=model_dir/eval
      ```

   2. Inference on image using [object_detection_tutorial.ipynb](research/object_detection/object_detection_tutorial.ipynb).

      ```bash
      jupyter notebook
      ```

      

6. ### Model Optimization (OpenVINO)

   1. Open the json file, change `"Postprocessor/Cast"` to `"Postprocessor/Cast_1"`.

      ```bash
      sudo gedit /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v1.14.json
      ## Edit file here ##
      ```

   2. Optimize Tensorflow model

      ```bash
      source /opt/intel/openvino/bin/setupvars.sh
      cd /opt/intel/openvino/deployment_tools/model_optimizer/
      # Optimize Tensorflow model
      python3 mo_tf.py \
      	--input_model ~/models/research/exported_graphs/0/frozen_inference_graph.pb \
      	--tensorflow_object_detection_api_pipeline_config ~/models/research/exported_graphs/0/pipeline.config \
      	--tensorflow_use_custom_operations_config extensions/front/tf/ssd_support_api_v1.14.json \
      	--output_dir ~/openvino_models/ir/FP16/public/mobilenet-ssd/
      ```

   3. Copy optimized model into ros_vino package.

      ```bash
      cp -r ~/openvino_models/ir/FP16/public/mobilenet-ssd/ ~/catkin_ws/src/ros_vino/models/FP16/
      ```

7. ### Model Inference on NCS2 / Intel CPU

   1. Create a `~/catkin_ws`. Clone [ros_vino](https://github.com/songshan0321/ros_vino.git) (the package written by myself) and [realsense-ros](https://github.com/IntelRealSense/realsense-ros.git) package then compile.

      ```bash
      mkdir -p ~/catkin_ws/src
      cd ~/catkin_ws/src/
      git clone https://github.com/songshan0321/ros_vino.git
      git clone https://github.com/IntelRealSense/realsense-ros.git
      cd ~/catkin_ws && catkin_make
      ```

   2. Source ROS and OpenVINO environment

      ```bash
      source /opt/ros/kinetic/setup.bash
      source ~/catkin_ws/devel/setup.bash
      source /opt/intel/openvino/bin/setupvars.sh
      ```

   3. Start inference on NCS2 using Realsense camera and show results on rviz. More detail of this can be found in [ros_vino](https://github.com/songshan0321/ros_vino.git).

      ```bash
      roslaunch ros_vino object_detection_ssd_realsense.launch
      ```

      For other image sources, run:

      ```bash
      roslaunch ros_vino object_detection_ssd.launch
      ```

      