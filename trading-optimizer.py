ultralytics version: 8.3.146
WARNING ⚠️ 'label_smoothing' is deprecated and will be removed in in the future.
Ultralytics 8.3.146 🚀 Python-3.9.14 torch-2.7.0+cu126 CUDA:0 (NVIDIA A100-SXM4-80GB, 81153MiB)
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
File /usr/lib64/python3.9/pathlib.py:1323, in Path.mkdir(self, mode, parents, exist_ok)
   1322 try:
-> 1323     self._accessor.mkdir(self, mode)
   1324 except FileNotFoundError:

FileNotFoundError: [Errno 2] No such file or directory: 'runs/detect/train5/weights'

During handling of the above exception, another exception occurred:

PermissionError                           Traceback (most recent call last)
Cell In[11], line 48
     39     model.val(
     40         data=data_file,
     41         imgsz=960,
   (...)
     44         agnostic_nms=False
     45     )
     47 if __name__ == "__main__":
---> 48     main()

Cell In[11], line 12, in main()
      8 data_file = "test_venv/data-Copy1.yaml"
     10 model = YOLO(old_model_path)
---> 12 model.train(
     13     data=data_file,
     14     epochs=300,
     15     imgsz=960,
     16     batch=8,
     17     cache=True,
     18     cls=1.8,
     19     box=7.0,
     20     mosaic=0.70,
     21     mixup=0.0,
     22     translate=0.10,
     23     fliplr=0.50,
     24     degrees=0.0,
     25     shear=0.0,
     26     scale=0.0,
     27     hsv_h=0.015,
     28     hsv_s=0.70,
     29     hsv_v=0.40,
     30     label_smoothing=0.05,
     31     optimizer="AdamW",
     32     lr0=0.001,
     33     lrf=0.1,
     34     weight_decay=0.001,
     35     close_mosaic=15,
     36     workers=8
     37 )
     39 model.val(
     40     data=data_file,
     41     imgsz=960,
   (...)
     44     agnostic_nms=False
     45 )

File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/ultralytics/engine/model.py:791, in Model.train(self, trainer, **kwargs)
    788 if args.get("resume"):
    789     args["resume"] = self.ckpt_path
--> 791 self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
    792 if not args.get("resume"):  # manually set model only if not resuming
    793     self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)

File /project/ai-for-trap-processing-now/venv_now/lib64/python3.9/site-packages/ultralytics/engine/trainer.py:134, in BaseTrainer.__init__(self, cfg, overrides, _callbacks)
    132 self.wdir = self.save_dir / "weights"  # weights dir
    133 if RANK in {-1, 0}:
--> 134     self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
    135     self.args.save_dir = str(self.save_dir)
    136     YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args

File /usr/lib64/python3.9/pathlib.py:1327, in Path.mkdir(self, mode, parents, exist_ok)
   1325     if not parents or self.parent == self:
   1326         raise
-> 1327     self.parent.mkdir(parents=True, exist_ok=True)
   1328     self.mkdir(mode, parents=False, exist_ok=exist_ok)
   1329 except OSError:
   1330     # Cannot rely on checking for EEXIST, since the operating system
   1331     # could give priority to other errors like EACCES or EROFS

File /usr/lib64/python3.9/pathlib.py:1323, in Path.mkdir(self, mode, parents, exist_ok)
   1319 """
   1320 Create a new directory at this given path.
   1321 """
   1322 try:
-> 1323     self._accessor.mkdir(self, mode)
   1324 except FileNotFoundError:
   1325     if not parents or self.parent == self:

PermissionError: [Errno 13] Permission denied: 'runs/detect/train5'
