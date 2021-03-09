import os

src_dir = "/media/db/hdd/obj_project/tensorflow115/20201019_xingpian_fangxiang"
checkpoint_num = "model/model.ckpt-134204"

PIPELINE_CONFIG_PATH = os.path.join(src_dir, "ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config")
TRAINED_CKPT_PREFIX = os.path.join(src_dir, checkpoint_num)
EXPORT_DIR = os.path.join(src_dir, "export")
input_shape = "1,480,640,3"
# input_shape = "1,480,854,3"
# input_shape = "1,640,640,3"


INPUT_TYPE = "image_tensor"
max_detections = 200

output_file = os.path.join(EXPORT_DIR, TRAINED_CKPT_PREFIX.split("-")[-1] + ".tflite")
graph_def_file = os.path.join(EXPORT_DIR, "tflite_graph.pb")
add_postprocessing_op = True

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
