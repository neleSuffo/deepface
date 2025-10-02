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

def group_faces_sequentially(input_dir, output_dir):
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
    # Store the path of the last image added to the current cluster
    current_cluster_last_image = os.path.join(input_dir, image_files[0]) 
    groups[current_person_id].append(current_cluster_last_image)

    print("⏳ Beginning sequential grouping...")
    with tqdm(total=len(image_files) - 1, desc="Grouping images sequentially") as pbar:
        for i in range(1, len(image_files)):
            curr_img_path = os.path.join(input_dir, image_files[i])
            
            try:
                result = DeepFace.verify(
                    img1_path=current_cluster_last_image, # Compare to the last image in the current group
                    img2_path=curr_img_path,
                    enforce_detection=False
                )

                if result["verified"] == True:
                    groups[current_person_id].append(curr_img_path)
                    # If it's a match, update the last image for the next comparison
                    current_cluster_last_image = curr_img_path 
                else:
                    current_person_id += 1
                    groups[current_person_id].append(curr_img_path)
                    # If it's a new person, the current image becomes the new last image
                    current_cluster_last_image = curr_img_path 

            except Exception as e:
                logging.warning(f"Skipping comparison between {current_cluster_last_image} and {curr_img_path}: {e}")
                current_person_id += 1
                groups[current_person_id].append(curr_img_path)
                current_cluster_last_image = curr_img_path
            
            pbar.update(1)

    # Save the clustered images
    for cluster_id, images in groups.items():
        person_dir = os.path.join(output_dir, f"person_{cluster_id}")
        os.makedirs(person_dir, exist_ok=True)
        for img_path in images:
            shutil.copy(img_path, person_dir)

    print(f"✅ Estimated number of unique people: {len(groups)}")
    return groups

def reassign_small_clusters(groups, input_dir, min_size=3):
    """
    Reassigns images from small clusters (less than min_size) to other clusters.
    
    Args:
        groups (dict): The initial dictionary of clusters.
        input_dir (str): The directory containing all face images.
        min_size (int): The minimum size a cluster must be to be considered 'valid'.
        
    Returns:
        dict: The updated dictionary of clusters.
    """
    small_clusters_ids = {cluster_id for cluster_id, images in groups.items() if len(images) < min_size}
    if not small_clusters_ids:
        print("No small clusters to reassign.")
        return groups

    print(f"🕵️‍♂️ Found {len(small_clusters_ids)} small clusters to reassign...")

    sorted_image_files = sorted(
        [fname for fname in os.listdir(input_dir) if fname.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0].split('_face')[0].split('_')[-1])
    )
    
    reassigned_images = {} # To keep track of which images have been moved

    for cluster_id in tqdm(sorted(list(small_clusters_ids)), desc="Reassigning small clusters"):
        images_to_reassign = groups[cluster_id]
        print(f"\n🔍 Small cluster {cluster_id} (size {len(images_to_reassign)}):")
        for img_path in images_to_reassign:
            print(f"  Image: {img_path}")
            if img_path in reassigned_images:
                print(f"    Already reassigned, skipping.")
                continue
            # Step 1: Check the previous frame chronologically
            img_filename = os.path.basename(img_path)
            try:
                current_frame_number = int(os.path.splitext(img_filename)[0].split('_face')[0].split('_')[-1])
            except (ValueError, IndexError):
                logging.warning(f"Could not parse frame number from filename: {img_filename}")
                continue
            prev_frame_number = current_frame_number - 1
            prev_img_path = None
            # Find the closest previous frame with a face
            prev_img_path = None
            img_index = None
            for idx, filename in enumerate(sorted_image_files):
                if os.path.join(input_dir, filename) == img_path:
                    img_index = idx
                    break
            if img_index is not None and img_index > 0:
                # Find the closest previous image
                prev_img_path = os.path.join(input_dir, sorted_image_files[img_index - 1])
            found_new_cluster = False
            if prev_img_path:
                for prev_cluster_id, prev_images in groups.items():
                    if prev_img_path in prev_images:
                        print(f"    .verify: {prev_img_path} vs {img_path}")
                        try:
                            result = DeepFace.verify(
                                img1_path=prev_img_path,
                                img2_path=img_path,
                                enforce_detection=False
                            )
                            print(f"      Result: {result}")
                            if result["verified"]:
                                print(f"      Decision: Assigned to cluster {prev_cluster_id}")
                                groups[prev_cluster_id].append(img_path)
                                reassigned_images[img_path] = True
                                found_new_cluster = True
                                break
                            else:
                                print(f"      Decision: Not assigned")
                        except Exception as e:
                            logging.warning(f"Failed to verify {img_path} with {prev_img_path}: {e}")
                if found_new_cluster:
                    continue
            # Step 2: If no match with the previous frame, check all other clusters
            if not found_new_cluster:
                candidate_clusters = {cid: imgs[0] for cid, imgs in groups.items() if len(imgs) >= min_size}
                for candidate_id, candidate_img_path in candidate_clusters.items():
                    print(f"    .verify: {candidate_img_path} vs {img_path}")
                    try:
                        result = DeepFace.verify(
                            img1_path=candidate_img_path,
                            img2_path=img_path,
                            enforce_detection=False
                        )
                        print(f"      Result: {result}")
                        if result["verified"]:
                            print(f"      Decision: Assigned to cluster {candidate_id}")
                            groups[candidate_id].append(img_path)
                            reassigned_images[img_path] = True
                            found_new_cluster = True
                            break
                        else:
                            print(f"      Decision: Not assigned")
                    except Exception as e:
                        logging.warning(f"Failed to verify {img_path} with {candidate_img_path}: {e}")
            if not found_new_cluster:
                print(f"    Decision: Remains in small cluster {cluster_id}")

    # Clean up the groups dictionary
    final_groups = {cid: images for cid, images in groups.items() if len(images) > 0}
    return final_groups

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Group faces in a folder using a sequential, chronological approach.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing cropped face images (filenames should be numeric).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save grouped images (default: input_dir/groups)")
    parser.add_argument("--blur_threshold", type=float, default=4, help="The minimum blur score for an image to be considered.")
    parser.add_argument("--min_size", type=int, default=100, help="The minimum image dimension (height or width) in pixels.")
    # Add a new argument for the final cluster size
    parser.add_argument("--min_cluster_size", type=int, default=3, help="The minimum number of images in a final cluster.")

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
    groups = group_faces_sequentially(
        input_dir=filtered_dir,
        output_dir=output_dir,
    )

    # 3. Reassign small clusters
    final_groups = reassign_small_clusters(
        groups, 
        filtered_dir, 
        min_size=args.min_cluster_size
    )
    
    # 4. Save the valid clusters
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster_id, images in final_groups.items():
        person_dir = os.path.join(output_dir, f"person_{cluster_id}")
        os.makedirs(person_dir, exist_ok=True)
        for img_path in images:
            shutil.copy(img_path, person_dir)

    print(f"✅ Final number of unique people after filtering: {len(final_groups)}")
    
    # 5. Clean up temporary directory
    if os.path.exists(temp_filtered_dir):
        shutil.rmtree(temp_filtered_dir)
    print("Cleaned up temporary directories.")
    
if __name__ == "__main__":
    main()