from deepface import DeepFace

#img1_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_annotated/clusters/cluster_8/quantex_at_home_id255944_2022_03_08_01_002300_face_1.PNG"
#img2_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_annotated/clusters/cluster_2/quantex_at_home_id255944_2022_03_08_01_002440_face_1.PNG"
#result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, enforce_detection=False, detector_backend = "retinaface")

img1_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_annotated/clusters/cluster_3/quantex_at_home_id255944_2022_03_08_01_002320_face_1.PNG"
objs = DeepFace.analyze(
  img_path = img1_path, actions = ['age', 'gender'], enforce_detection=False, detector_backend = "retinaface"
)

#face_objs = DeepFace.extract_faces(
#  img_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_annotated/quantex_at_home_id255944_2022_03_08_01_000090_face_1.PNG", detector_backend = "yolov12l", align = True, enforce_detection = False
#)

print(objs)
# if result["verified"]:
#     print("Faces match!")
# else:
#     print("Faces do not match.")