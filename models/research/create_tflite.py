import os

# os.system("python inference.py --data_dir=%s --save_dir=%s --GPU=%s" % ("../demos/", "../inference_results/","0"))
# os.system("python object_detection/dataset_tools/create_pascal_tf_record.py --label_map_path=%s --data_dir=%s --year=%s --set=%s --output_path=%s" % ("object_detection/data/pascal_label_map.pbtxt","VOCdevkit","VOC2012","train","pascal_train.record"))
# os.system("python object_detection/dataset_tools/create_pascal_tf_record.py --label_map_path=%s --data_dir=%s --year=%s --set=%s --output_path=%s" % ("object_detection/data/pascal_label_map.pbtxt","VOCdevkit","VOC2012","val","pascal_val.record"))
# os.system("python object_detection/dataset_tools/create_pascal_tf_record.py --label_map_path=%s --data_dir=%s --output_path=%s" % ("object_detection/data/pascal_label_map.pbtxt","example","pascal_train.record"))


# PIPELINE_CONFIG_PATH = "/home/sucom/dbing/tensor_offical/models-master/research/object_detection/protos/pipeline.proto"
# MODEL_DIR = "/home/sucom/Desktop/model"
# NUM_TRAIN_STEPS = 50000
# SAMPLE_1_OF_N_EVAL_EXAMPLES = 1


# pipeline_config_path = "/home/sucom/Documents/led_lamp_0610/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18/pipeline.config"
pipeline_config_path = "/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"

trained_checkpoint_prefix = "/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/model/model.ckpt-384143"
output_directory = "/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/frozen"
add_postprocessing_op = True
max_detections = 200
#
os.system(
    "python object_detection/export_tflite_ssd_graph.py --max_detections=%s --pipeline_config_path=%s --trained_checkpoint_prefix=%s --output_directory=%s --add_postprocessing_op=%s" % (
    max_detections,pipeline_config_path, trained_checkpoint_prefix, output_directory, add_postprocessing_op))

# input_shape = "1,512,640,3"
input_shape = "1,480,640,3"
# input_shape = "1,360,480,3"
output_file = '/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/frozen/384143.tflite'
graph_def_file = output_directory + "/tflite_graph.pb"

os.system("tflite_convert \
  --input_shape=%s \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --allow_custom_ops \
  --graph_def_file=%s \
  --inference_type=FLOAT\
  --output_file=%s" % (input_shape, graph_def_file, output_file))

# os.system("python object_detection/model_main.py")
