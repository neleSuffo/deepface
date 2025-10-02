from deepface import DeepFace

img1_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255237_2022_05_08_03_051960_face_1.png"
img2_path = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id257147_2021_10_31_03_000670_face_1.png"
result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, enforce_detection=False)

print(result)