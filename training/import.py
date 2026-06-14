from roboflow import Roboflow
rf = Roboflow(api_key="TZ17MB6LRrkHFDnteWDk")
project = rf.workspace("tigers-domain").project("rose-diseases-detection-rryri")
version = project.version(3)
dataset = version.download("yolo26")
                