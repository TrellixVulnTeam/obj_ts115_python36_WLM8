import os

INPUT_TYPE="image_tensor"
PIPELINE_CONFIG_PATH="/home/sucom/Documents/20201109_854_480_tuqi/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"
TRAINED_CKPT_PREFIX="/home/sucom/Documents/20201109_854_480_tuqi/model/model.ckpt-37962"
EXPORT_DIR="/home/sucom/Documents/20201109_854_480_tuqi/frozen"



os.system(
    "python object_detection/export_inference_graph.py --input_type=%s --pipeline_config_path=%s --trained_checkpoint_prefix=%s --output_directory=%s " % (
        INPUT_TYPE, PIPELINE_CONFIG_PATH, TRAINED_CKPT_PREFIX, EXPORT_DIR))

"""

tensorflowjs_converter --input_format = tf_frozen_model \
                       --output_format = tfjs_graph_model \
                       --output_node_names = ' Postprocessor / ExpandDims_1,Postprocessor / Slice ' \
                       ./frozen_inference_graph.pb \
                       ./web_model

'TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'

tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='detection_boxes:0','detection_scores:0','detection_classes:0','num_detections:0' \
    /home/sucom/Desktop/luoding_self/ld_self/js/frozen_inference_graph.pb \
    /home/sucom/Desktop/luoding_self/ld_self/js/saved_model/web_model

"""

