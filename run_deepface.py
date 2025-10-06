from deepface import DeepFace

img1_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_annotated/groups/person_3/quantex_at_home_id255944_2022_03_08_01_011530_face_1.PNG"
img2_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_annotated/groups/person_4/quantex_at_home_id255944_2022_03_08_01_011650_face_1.PNG"
result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, enforce_detection=False)

if result["verified"]:
    print("Faces match!")
else:
    print("Faces do not match.")
