Traceback (most recent call last):

  File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/IPython/core/interactiveshell.py:3550 in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)

  Cell In[10], line 50
    main()

  Cell In[10], line 9 in main
    model.train(

  File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/ultralytics/engine/model.py:791 in train
    self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)

  File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/ultralytics/engine/trainer.py:119 in __init__
    self.args = get_cfg(cfg, overrides)

  File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/ultralytics/cfg/__init__.py:305 in get_cfg
    check_dict_alignment(cfg, overrides)

  File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/ultralytics/cfg/__init__.py:498 in check_dict_alignment
    raise SyntaxError(string + CLI_HELP_MSG) from e

  File <string>
SyntaxError: 'fl_gamma' is not a valid YOLO argument. 

    Arguments received: ['yolo', '-f', '/home/zachary.dawson/.local/share/jupyter/runtime/kernel-7bd40716-dc55-4c64-9e3a-9114d9c2e6a4.json']. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of ['classify', 'segment', 'pose', 'detect', 'obb']
                MODE (required) is one of ['train', 'track', 'predict', 'benchmark', 'export', 'val']
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    5. Ultralytics solutions usage
        yolo solutions count or in ['crop', 'blur', 'workout', 'heatmap', 'isegment', 'visioneye', 'speed', 'queue', 'analytics', 'inference', 'trackzone'] source="path/to/video.mp4"

    6. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help

    Docs: https://docs.ultralytics.com
    Solutions: https://docs.ultralytics.com/solutions/
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    
