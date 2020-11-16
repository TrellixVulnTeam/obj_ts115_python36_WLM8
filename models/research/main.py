import os

# os.system("python inference.py --data_dir=%s --save_dir=%s --GPU=%s" % ("../demos/", "../inference_results/","0"))
# os.system("python object_detection/dataset_tools/create_pascal_tf_record.py --label_map_path=%s --data_dir=%s --year=%s --set=%s --output_path=%s" % ("object_detection/data/pascal_label_map.pbtxt","VOCdevkit","VOC2012","train","pascal_train.record"))
# os.system("python object_detection/dataset_tools/create_pascal_tf_record.py --label_map_path=%s --data_dir=%s --year=%s --set=%s --output_path=%s" % ("object_detection/data/pascal_label_map.pbtxt","VOCdevkit","VOC2012","val","pascal_val.record"))
# os.system("python object_det   ection/dataset_tools/create_pascal_tf_record.py --label_map_path=%s --data_dir=%s --output_path=%s" % ("object_detection/data/pascal_label_map.pbtxt","example","pascal_train.record"))


# PIPELINE_CONFIG_PATH = "/home/sucom/dbing/tensor_offical/models-master/research/object_detection/protos/pipeline.proto"
# MODEL_DIR = "/home/sucom/Desktop/model"
# NUM_TRAIN_STEPS = 50000
# SAMPLE_1_OF_N_EVAL_EXAMPLES = 1


PIPELINE_CONFIG_PATH = "/home/db/桌面/目标检测项目/ts114_py36/20201113_water/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"

MODEL_DIR = "/home/db/桌面/目标检测项目/ts114_py36/20201113_water/model"
# MODEL_DIR = "/home/db/桌面/目标检测项目/ts114_py36/lsd_all_test_20201110/model"
# MODEL_DIR = "/home/sucom/Documents/5720_led_20201023/model"

NUM_TRAIN_STEPS = 400000000
SAMPLE_1_OF_N_EVAL_EXAMPLES = 1

os.system("python object_detection/model_main.py --pipeline_config_path=%s --model_dir=%s --num_train_steps=%s --sample_1_of_n_eval_examples=%s --alsologtostderr" % (PIPELINE_CONFIG_PATH,MODEL_DIR,NUM_TRAIN_STEPS,SAMPLE_1_OF_N_EVAL_EXAMPLES))
# os.system("python object_detectionf/model_main.py")

