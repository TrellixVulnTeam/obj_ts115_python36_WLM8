import os

INPUT_TYPE = "image_tensor"
PIPELINE_CONFIG_PATH = "/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"
TRAINED_CKPT_PREFIX = "/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/model/model.ckpt-384143"
EXPORT_DIR = "/media/db/WLZ_Secret_db/训练的模型文件和记录/5720/放飞/飞机头部/export"
input_shape = "1,480,640,3"


output_file = os.path.join(EXPORT_DIR, TRAINED_CKPT_PREFIX.split("-")[-1] + ".tflite")
graph_def_file = os.path.join(EXPORT_DIR, "tflite_graph.pb")
add_postprocessing_op = True
max_detections = 200

os.system(
    "python object_detection/export_tflite_ssd_graph.py --max_detections=%s --pipeline_config_path=%s --trained_checkpoint_prefix=%s --output_directory=%s --add_postprocessing_op=%s" % (
    max_detections, PIPELINE_CONFIG_PATH, TRAINED_CKPT_PREFIX, EXPORT_DIR, add_postprocessing_op))

os.system("tflite_convert \
  --input_shape=%s \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --allow_custom_ops \
  --graph_def_file=%s \
  --inference_type=FLOAT\
  --output_file=%s" % (input_shape, graph_def_file, output_file))

os.system(
    "python object_detection/export_inference_graph.py --input_type=%s --pipeline_config_path=%s --trained_checkpoint_prefix=%s --output_directory=%s " % (
        INPUT_TYPE, PIPELINE_CONFIG_PATH, TRAINED_CKPT_PREFIX, EXPORT_DIR))
