from ultralitics import YOLO 

model = YOLO("")

resaults = model(source = 1, show = true , conf = 0.4, save = true)
