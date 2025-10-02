import os
import argparse
import shutil
import logging
import cv2
from deepface import DeepFace
from tqdm import tqdm
from collections import defaultdict

# --- Utility Functions ---

def calculate_blur_score(image_path):
    """Calculates a blur score for an image using the Laplacian variance."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return float('inf')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        logging.warning(f"Could not calculate blur score for {image_path}: {e}")
        return float('inf')

def filter_images(input_dir, output_dir, blur_threshold, min_size_pixels):
    """
    Filters out images based on blur score and size.
    Returns the path to the directory containing the filtered images.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"🔬 Found {len(image_files)} images to filter...")
    
    filtered_count = 0
    for filename in tqdm(image_files, desc="Filtering images by blur and size"):
        image_path = os.path.join(input_dir, filename)
        
        # Check size
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w, _ = image.shape
            if h < min_size_pixels or w < min_size_pixels:
                continue # Skip small images
        except Exception as e:
            logging.warning(f"Could not read image dimensions for {image_path}: {e}")
            continue

        # Check blur
        score = calculate_blur_score(image_path)
        
        if score > blur_threshold:
            shutil.copy(image_path, os.path.join(output_dir, filename))
            filtered_count += 1
    
    print(f"✅ Filtered {len(image_files) - filtered_count} images. {filtered_count} images remain.")
    return output_dir

def group_faces_sequentially(input_dir, output_dir, model_name="Facenet"):
    """
    Groups face images by comparing each image to the one that chronologically precedes it.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(
        [fname for fname in os.listdir(input_dir) if fname.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0].split('_face')[0].split('_')[-1])
    )
    logging.info(f"Found {len(image_files)} face images to process.")

    if not image_files:
        print("No images to process.")
        return

    groups = defaultdict(list)
    current_person_id = 0
    groups[current_person_id].append(os.path.join(input_dir, image_files[0]))

    print("⏳ Beginning sequential grouping...")
    with tqdm(total=len(image_files) - 1, desc="Grouping images sequentially") as pbar:
        for i in range(1, len(image_files)):
            prev_img_path = os.path.join(input_dir, image_files[i-1])
            curr_img_path = os.path.join(input_dir, image_files[i])
            
            try:
                result = DeepFace.verify(
                    img1_path=prev_img_path,
                    img2_path=curr_img_path,
                    model_name=model_name,
                    enforce_detection=False
                )

                if result["verified"] == True:
                    groups[current_person_id].append(curr_img_path)
                else:
                    current_person_id += 1
                    groups[current_person_id].append(curr_img_path)

            except Exception as e:
                logging.warning(f"Skipping comparison between {prev_img_path} and {curr_img_path}: {e}")
                # If an error occurs, we assume it's a new person to be safe
                current_person_id += 1
                groups[current_person_id].append(curr_img_path)
            
            pbar.update(1)

    # Save the clustered images
    for cluster_id, images in groups.items():
        person_dir = os.path.join(output_dir, f"person_{cluster_id}")
        os.makedirs(person_dir, exist_ok=True)
        for img_path in images:
            shutil.copy(img_path, person_dir)

    print(f"✅ Estimated number of unique people: {len(groups)}")

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Group faces in a folder using a sequential, chronological approach.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing cropped face images (filenames should be numeric).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save grouped images (default: input_dir/groups)")
    parser.add_argument("--model_name", type=str, default="Facenet", choices=["Facenet", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"], help="DeepFace model to use for verification")
    parser.add_argument("--blur_threshold", type=float, default=4, help="The minimum blur score for an image to be considered.")
    parser.add_argument("--min_size", type=int, default=100, help="The minimum image dimension (height or width) in pixels.")

    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, "groups")
    temp_filtered_dir = os.path.join(os.path.dirname(output_dir), "temp_filtered_faces")
    
    # 1. Filter out blurry and small images
    print("Beginning image filtering process...")
    filtered_dir = filter_images(args.input_dir, temp_filtered_dir, args.blur_threshold, args.min_size)
    
    if filtered_dir is None:
        print("Exiting due to an error in the filtering process.")
        return

    # 2. Cluster the filtered images using the sequential approach
    print("\nBeginning face grouping process...")
    group_faces_sequentially(
        input_dir=filtered_dir,
        output_dir=output_dir,
        model_name=args.model_name
    )
    
    # 3. Clean up temporary directory
    if os.path.exists(temp_filtered_dir):
        shutil.rmtree(temp_filtered_dir)
    print("Cleaned up temporary directories.")

if __name__ == "__main__":
    main()