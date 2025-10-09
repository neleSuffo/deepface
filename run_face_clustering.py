import cv2
import csv
import json
import re
import argparse
import logging
import shutil
from deepface import DeepFace
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from config import FaceConfig

# -------------------
# Utilities
# -------------------
_verify_cache = {}

def calculate_blur_score(image_path):
    """Calculates a blur score for an image using the Laplacian variance.
    A lower score indicates a blurrier image.
    
    Parameters:
    ----------
    image_path (str): 
        Path to the image file.
        
    Returns:
    -------
    float:
        Variance of the Laplacian; lower means blurrier.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return float('-inf')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        logging.warning(f"blur calc failed for {image_path}: {e}")
        return float('-inf')


def get_image_size(image_path):
    """Returns image area (width * height).
    
    Parameters:
    ----------
    image_path (str):
        Path to the image file.
    
    Returns: 
    -------
    int:
        Image area in pixels (width * height), or 0 on failure.    
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        h, w = img.shape[:2]
        return int(h * w)
    except Exception as e:
        logging.warning(f"size calc failed for {image_path}: {e}")
        return 0


def extract_frame_number(filename):
    """Extracts a 6-digit frame number from the filename, or None if not found. 
    
    Parameters:
    ----------
    filename (Path):
        Path object representing the image file.
        
    Returns: 
    -------
    int or None:
        Extracted frame number, or None if not found.
    """
    basename = filename.name
    match = re.search(r'_(\d{6})_face', basename)
    if match:
        return int(match.group(1))
    return None


def verify_pair(img1, img2):
    """Function to verify if two images are of the same person using DeepFace.
    Caches results to avoid redundant computations.
    
    Parameters:
    ----------
    img1 (str or Path):
        Path to the first image file.
    img2 (str or Path):
        Path to the second image file. 
        
    Returns:
    -------
    (bool, float or None):
        Tuple of (verified, distance). 'verified' is True if images match, False otherwise.
        'distance' is the similarity distance, or None if verification failed.
    """
    # symmetric key
    key = tuple(sorted([str(img1), str(img2)]))
    if key in _verify_cache:
        return _verify_cache[key]

    try:
        res = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
        verified = bool(res.get("verified", False))
        distance = res.get("distance", None)
    except Exception as e:
        logging.warning(f"DeepFace.verify failed for {img1} vs {img2}: {e}")
        verified, distance = False, None

    _verify_cache[key] = (verified, distance)
    return verified, distance


def compute_best_representative(image_list):
    """Return best image path (sharpest then largest) from a list, or None if empty."""
    if not image_list:
        return None
    best = None
    for p in image_list:
        b = calculate_blur_score(p)
        s = get_image_size(p)
        if best is None or s > best["size"] or (s == best["size"] and b > best["blur"]):
            best = {"path": p, "blur": b, "size": s}
    return best


# -------------------
# Clustering (deferred assignment of ambiguous frames)
# -------------------
def cluster_faces(image_paths):
    """
    Core algorithm:
      - Uses a quality threshold to select the first cluster image.
      - Compares incoming images to the cluster's BEST representative.
      - Never finalize ambiguous frame assignments until buffer resolution.
    
    Returns:
      clusters: dict cluster_id -> [image_paths]
      representatives: dict cluster_id -> {'path','blur','size'}
      assignments: dict image_path -> {cluster_id, compared_with, verified, distance}
    """
    clusters = defaultdict(list)
    representatives = {}
    assignments = {}  # final assignment map (image -> metadata)
    ambiguous = []  # buffer of ambiguous images (not yet assigned)

    # Find the first sufficiently sharp image to start the first cluster ---
    first_sharp_idx = -1
    for i, path in enumerate(image_paths):
        blur = calculate_blur_score(str(path))
        if blur > FaceConfig.REPRESENTATIVE_BLUR_THRESHOLD:
            first_sharp_idx = i
            break
    
    if first_sharp_idx == -1:
        logging.warning("No sharp enough images found to start clustering.")
        return clusters, representatives, assignments

    # Init with the first sharp image as cluster 1
    current_cluster = 1
    first = image_paths[first_sharp_idx]
    clusters[current_cluster].append(first)
    rep = compute_best_representative([first])
    representatives[current_cluster] = rep
    assignments[first] = {
        "cluster_id": current_cluster,
        "compared_with": None,
        "verified": True,
        "distance": None
    }

    # assign first image metadata
    last_in_cluster = first

    # Start loop from the image immediately following the one used to start the cluster
    for idx in range(first_sharp_idx + 1, len(image_paths)):
        cur = image_paths[idx]

        # Compare incoming image to the current cluster's BEST representative
        rep_path = representatives[current_cluster]["path"]
        verified, distance = verify_pair(rep_path, cur)

        if verified:
            # incoming matches current cluster representative -> everything in ambiguous belongs to current cluster
            if ambiguous:
                # Assign ambiguous frames back to the current cluster, comparing against the *new* good image (cur)
                # or the representative. Comparing against 'cur' is often better as it's closer in time.
                for amb in ambiguous:
                    # v, d = verify_pair(rep_path, amb) # Alternative: compare to representative
                    v, d = verify_pair(cur, amb) # Compare against the newly verified, close-in-time image
                    
                    # Assume it should be assigned if 'cur' verified against the rep (strong signal)
                    clusters[current_cluster].append(amb)
                    assignments[amb] = {
                        "cluster_id": current_cluster,
                        "compared_with": cur, 
                        "verified": bool(v),
                        "distance": d
                    }
                    # update representative if necessary
                    rep_curr = representatives[current_cluster]
                    b = calculate_blur_score(amb); s = get_image_size(amb)
                    if b > rep_curr["blur"] or (b == rep_curr["blur"] and s > rep_curr["size"]):
                        representatives[current_cluster] = {"path": amb, "blur": b, "size": s}
                ambiguous = []

            # assign current image to current cluster
            clusters[current_cluster].append(cur)
            assignments[cur] = {
                "cluster_id": current_cluster,
                "compared_with": rep_path, # Log comparison against the representative
                "verified": True,
                "distance": distance
            }
            # update representative if needed
            rep_curr = representatives[current_cluster]
            b = calculate_blur_score(cur); s = get_image_size(cur)
            if b > rep_curr["blur"] or (b == rep_curr["blur"] and s > rep_curr["size"]):
                representatives[current_cluster] = {"path": cur, "blur": b, "size": s}
            last_in_cluster = cur  # update last image in cluster for potential future use (and tie-break at end)
            continue

        # not verified -> buffer it
        ambiguous.append(cur)
        
        # if buffer reaches threshold -> decide
        if len(ambiguous) >= FaceConfig.CLUSTER_CONSECUTIVE_FRAMES:
            # probe = last ambiguous frame
            probe = ambiguous[-1]

            # 1) try matching probe against other cluster representatives
            matched_cluster = None
            matched_rep_path = None
            matched_distance = None
            for cid, repinfo in representatives.items():
                if cid == current_cluster:
                    continue
                v_rep, d_rep = verify_pair(repinfo["path"], probe)
                if v_rep:
                    matched_cluster = cid
                    matched_rep_path = repinfo["path"]
                    matched_distance = d_rep
                    break

            if matched_cluster is not None:
                # assign all ambiguous to the matched existing cluster
                for amb in ambiguous:
                    v, d = verify_pair(matched_rep_path, amb)
                    clusters[matched_cluster].append(amb)
                    assignments[amb] = {
                        "cluster_id": matched_cluster,
                        "compared_with": matched_rep_path,
                        "verified": bool(v),
                        "distance": d
                    }
                    # update representative for matched_cluster if needed
                    rep_curr = representatives[matched_cluster]
                    b = calculate_blur_score(amb); s = get_image_size(amb)
                    if b > rep_curr["blur"] or (b == rep_curr["blur"] and s > rep_curr["size"]):
                        representatives[matched_cluster] = {"path": amb, "blur": b, "size": s}
                # move current pointer to matched_cluster
                current_cluster = matched_cluster
                # last_in_cluster update is implicitly the last image of the merged cluster
                last_in_cluster = clusters[current_cluster][-1] 
                ambiguous = []
                continue

            # 2) else -> create a new cluster with all ambiguous frames (including probe)
            current_cluster += 1
            clusters[current_cluster].extend(ambiguous)
            # compute representative from ambiguous frames
            best_rep = compute_best_representative(ambiguous)
            if best_rep is None:
                # fallback: pick last ambiguous
                best_rep = {"path": ambiguous[-1], "blur": calculate_blur_score(ambiguous[-1]), "size": get_image_size(ambiguous[-1])}
            representatives[current_cluster] = best_rep

            # assign metadata for each ambiguous frame 
            for amb in ambiguous:
                v, d = verify_pair(best_rep["path"], amb)
                # clusters[current_cluster].append # already extended
                assignments[amb] = {
                    "cluster_id": current_cluster,
                    "compared_with": best_rep["path"],
                    "verified": bool(v),
                    "distance": d
                }
            last_in_cluster = ambiguous[-1]  # last image in new cluster
            ambiguous = []
            continue

    # End loop: if any ambiguous frames remain check each of them against all cluster representatives
    # and assign to the first matching cluster (if any)
    if ambiguous:
        for amb in ambiguous:
            # compare to all representatives
            assigned = False
            for cid, repinfo in representatives.items():
                v, d = verify_pair(repinfo["path"], amb)
                if v:
                    clusters[cid].append(amb)
                    assignments[amb] = {
                        "cluster_id": cid,
                        "compared_with": repinfo["path"],
                        "verified": True,
                        "distance": d
                    }
                    # update representative if needed
                    rep_curr = representatives[cid]
                    b = calculate_blur_score(amb); s = get_image_size(amb)
                    if b > rep_curr["blur"] or (b == rep_curr["blur"] and s > rep_curr["size"]):
                        representatives[cid] = {"path": amb, "blur": b, "size": s}
                    assigned = True
                    break
            if not assigned:
                # tie-break: assign to current cluster (original policy kept)
                clusters[current_cluster].append(amb)
                assignments[amb] = {
                    "cluster_id": current_cluster,
                    "compared_with": representatives[current_cluster]["path"],
                    "verified": False,
                    "distance": None
                }
                # update representative if needed
                rep_curr = representatives[current_cluster]
                b = calculate_blur_score(amb); s = get_image_size(amb)
                if b > rep_curr["blur"] or (b == rep_curr["blur"] and s > rep_curr["size"]):
                    representatives[current_cluster] = {"path": amb, "blur": b, "size": s}
                    
        # update representative based on added images
        rep_curr = representatives[current_cluster]
        new_rep = compute_best_representative(clusters[current_cluster])
        if new_rep and new_rep["path"] != rep_curr["path"]:
            representatives[current_cluster] = new_rep

    return clusters, representatives, assignments


def save_faces_to_clusters(image_paths, cluster_labels, output_dir):
    """
    Copies each face image into its respective cluster folder.
    Args:
        image_paths (list): List of image file paths.
        cluster_labels (list or array): Cluster assignment for each image.
        output_dir (str or Path): Output directory for clusters.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id in set(cluster_labels):
        cluster_folder = output_dir / f"cluster_{cluster_id}"
        cluster_folder.mkdir(parents=True, exist_ok=True)
    for img_path, label in tqdm(zip(image_paths, cluster_labels), total=len(image_paths), desc="Copying images to cluster folders"):
        cluster_folder = output_dir / f"cluster_{label}"
        shutil.copy2(img_path, cluster_folder)

# -------------------
# Run & export
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster face images in a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing face images.")
    args = parser.parse_args()

    IMG_DIR = Path(args.input_dir)
    CLST_DIR = IMG_DIR/"clusters"
    CLST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV = CLST_DIR/"face_clusters.csv"
    OUTPUT_JSON = CLST_DIR/"face_clusters.json"
    
    image_paths = sorted(
        [f for f in IMG_DIR.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg") and get_image_size(str(f)) >= 10000 and calculate_blur_score(str(f)) > FaceConfig.MIN_BLUR_THRESHOLD]
    )
    if not image_paths:
        raise ValueError(f"No images in {IMG_DIR}")

    start = datetime.now()
    clusters, reps, assignments = cluster_faces(image_paths)

    # Build final ordered results (one row per input image, in same order)
    results = []
    for p in image_paths:
        meta = assignments.get(p)
        frame_no = extract_frame_number(p)
        rep_path = None
        if meta and meta["cluster_id"] in reps:
            rep_path = reps[meta["cluster_id"]]["path"]
        results.append({
            "frame": frame_no,
            "filename": p.name,
            "cluster_id": meta["cluster_id"] if meta else -1,
            "compared_with": meta["compared_with"].name if meta and meta["compared_with"] else None,
            "verified": meta["verified"] if meta else False,
            "distance": meta["distance"] if meta else None,
            "representative": str(rep_path) if rep_path else None
        })

    # write CSV + JSON
    with open(OUTPUT_CSV, "w", newline="") as cf:
        fieldnames = ["frame", "filename", "cluster_id", "compared_with", "verified", "distance", "representative"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(OUTPUT_JSON, "w") as jf:
        json.dump(results, jf, indent=2)

    print(f"Clusters: {len(clusters)}  —  Saved to {OUTPUT_CSV}, {OUTPUT_JSON}")
    print("Elapsed (s):", (datetime.now() - start).total_seconds())

    # Print summary of clusters, images per cluster and representatives per cluster
    print("\nCluster Summary:")
    for cid, imgs in clusters.items():
        print(f"  Cluster {cid}: {len(imgs)} images")
        rep = reps.get(cid)
        if rep:
            print(f"    Representative: {Path(rep['path']).name}")

    # Build a mapping from image path to cluster id
    img_to_cluster = {}
    for cid, imgs in clusters.items():
        for img in imgs:
            img_to_cluster[img] = cid
    # Get cluster labels for each image (default to -1 if not found)
    cluster_labels = [img_to_cluster.get(img, -1) for img in image_paths]
    save_faces_to_clusters(image_paths, cluster_labels, CLST_DIR)