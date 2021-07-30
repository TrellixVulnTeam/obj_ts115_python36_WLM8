import os


PIPELINE_CONFIG_PATH = "/media/db/hdd2/obj_project/tensorflow115/LH训练完成模型检查点/20210618_171_9/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"
# PIPELINE_CONFIG_PATH = "/home/db/桌面/目标检测项目/ts114_py36/lsd_all_test_20201110/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"
# PIPELINE_CONFIG_PATH = "/home/db/桌面/目标检测项目/ts114_py36/20201102_red_s/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config"


MODEL_DIR = "/media/db/hdd2/obj_project/tensorflow115/LH训练完成模型检查点/20210618_171_9/model"
# MODEL_DIR = "/home/db/桌面/目标检测项目/ts114_py36/20201102_red_s/model"
# MODEL_DIR = "/home/db/桌面/目标检测项目/ts114_py36/5720_led_20201023/model"

NUM_TRAIN_STEPS = 400000000
SAMPLE_1_OF_N_EVAL_EXAMPLES = 1

os.system("python object_detection/model_main.py --pipeline_config_path=%s --model_dir=%s --num_train_steps=%s --sample_1_of_n_eval_examples=%s --alsologtostderr" % (PIPELINE_CONFIG_PATH,MODEL_DIR,NUM_TRAIN_STEPS,SAMPLE_1_OF_N_EVAL_EXAMPLES))
# os.system("python object_detectionf/model_main.py")
