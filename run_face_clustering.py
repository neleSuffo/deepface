import os
import cv2
import csv
import json
import re
import logging
from deepface import DeepFace
from collections import defaultdict
from datetime import datetime

# -------------------
# Configuration
# -------------------
IMG_DIR = "/home/nele_pauline_suffo/outputs/face_detections/quantex_at_home_id255944_2022_03_08_01_enhanced_annotated"
OUTPUT_CSV = os.path.join(IMG_DIR, "face_clusters.csv")
OUTPUT_JSON = os.path.join(IMG_DIR, "face_clusters.json")

CON_FRAMES = 3           # number of consecutive non-matching frames before starting new cluster
ENFORCE_DETECTION = False
# -------------------
# Utility Functions
# -------------------
def calculate_blur_score(image_path):
    """Calculates blur score using Laplacian variance (higher = sharper)."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return float('-inf')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        logging.warning(f"Could not calculate blur score for {image_path}: {e}")
        return float('-inf')


def get_image_size(image_path):
    """Returns total pixel count (width × height)."""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    return h * w


def extract_frame_number(filename):
    """Extracts the 6-digit frame number from filename like ..._000040_face_1."""
    basename = os.path.basename(filename)
    match = re.search(r'_(\d{6})_face', basename)
    if match:
        return int(match.group(1))
    return None


# -------------------
# Verification with caching
# -------------------
_verify_cache = {}
def verify_pair(img1, img2):
    """
    Wrapper around DeepFace.verify that caches results.
    Returns: (verified: bool, distance: float or None)
    """
    key = tuple(sorted([str(img1), str(img2)]))
    if key in _verify_cache:
        return _verify_cache[key]

    try:
        res = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=ENFORCE_DETECTION)
        verified = bool(res.get("verified", False))
        distance = res.get("distance", None)
    except Exception as e:
        logging.warning(f"DeepFace.verify failed for {img1} vs {img2}: {e}")
        verified, distance = False, None

    _verify_cache[key] = (verified, distance)
    return verified, distance


def compute_best_representative(image_paths):
    """Return representative dict {'path','blur','size'} from list (largest+sharpest)."""
    best = None
    for p in image_paths:
        b = calculate_blur_score(p)
        s = get_image_size(p)
        if best is None or b > best["blur"] or (b == best["blur"] and s > best["size"]):
            best = {"path": p, "blur": b, "size": s}
    return best


# -------------------
# Clustering Logic (fixed)
# -------------------
def cluster_faces(image_paths):
    """
    Clusters sequential face crops with the logic:
      - compare new image to last confirmed image of current cluster first
      - keep ambiguous_images until resolved or until CON_FRAMES => new cluster
      - check ambiguous images against existing representatives before creating new cluster
      - representative = largest + sharpest image for that cluster
    Returns:
      - clusters: dict cluster_id -> [image_paths]
      - representatives: dict cluster_id -> {'path','blur','size'}
      - assignments: dict image_path -> {cluster_id, compared_with, verified, distance}
    """
    clusters = defaultdict(list)
    representatives = {}
    assignments = {}  # img_path -> dict with metadata

    # init first cluster with first image
    current_cluster = 1
    first = image_paths[0]
    clusters[current_cluster].append(first)
    representatives[current_cluster] = compute_best_representative([first])
    assignments[first] = {
        "cluster_id": current_cluster,
        "compared_with": None,
        "verified": True,
        "distance": None
    }

    ambiguous_images = []  # images that mismatched last_confirmed and are pending resolution
    consecutive_non_matches = 0
    # last_confirmed_img for current_cluster is the most recent image we appended as confirmed
    last_confirmed_img = first

    for i in range(1, len(image_paths)):
        cur = image_paths[i]

        # ALWAYS compare current image to the last confirmed image of the current cluster first
        verified_to_last, dist_to_last = verify_pair(last_confirmed_img, cur)

        if verified_to_last:
            # If it matches the last confirmed, assign ALL ambiguous images (if any) + cur to current cluster
            if ambiguous_images:
                # mark all ambiguous images as belonging to current_cluster (they must have been similar to last_confirmed)
                for amb in ambiguous_images:
                    clusters[current_cluster].append(amb)
                    assignments[amb] = {
                        "cluster_id": current_cluster,
                        "compared_with": last_confirmed_img,
                        "verified": True,
                        "distance": dist_to_last  # we don't have their specific distance; this indicates resolution via last_confirmed
                    }
                    # potentially update representative
                    rep = representatives[current_cluster]
                    b = calculate_blur_score(amb); s = get_image_size(amb)
                    if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                        representatives[current_cluster] = {"path": amb, "blur": b, "size": s}
                ambiguous_images = []

            # assign cur itself
            clusters[current_cluster].append(cur)
            assignments[cur] = {
                "cluster_id": current_cluster,
                "compared_with": last_confirmed_img,
                "verified": True,
                "distance": dist_to_last
            }
            # update representative if needed
            rep = representatives[current_cluster]
            b = calculate_blur_score(cur); s = get_image_size(cur)
            if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                representatives[current_cluster] = {"path": cur, "blur": b, "size": s}

            last_confirmed_img = cur
            consecutive_non_matches = 0
            continue

        # NOT matched to last_confirmed -> ambiguous
        ambiguous_images.append(cur)
        consecutive_non_matches += 1

        # Try to resolve ambiguous images by checking if any ambiguous match last_confirmed
        matched_to_current = False
        for amb in ambiguous_images:
            v, d = verify_pair(last_confirmed_img, amb)
            if v:
                # If any ambiguous matches last_confirmed, add all ambiguous to current cluster
                for a in ambiguous_images:
                    clusters[current_cluster].append(a)
                    assignments[a] = {
                        "cluster_id": current_cluster,
                        "compared_with": last_confirmed_img,
                        "verified": True,
                        "distance": d
                    }
                    # update rep
                    rep = representatives[current_cluster]
                    b = calculate_blur_score(a); s = get_image_size(a)
                    if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                        representatives[current_cluster] = {"path": a, "blur": b, "size": s}
                last_confirmed_img = ambiguous_images[-1]
                ambiguous_images = []
                consecutive_non_matches = 0
                matched_to_current = True
                break
        if matched_to_current:
            continue

        # If none matched last_confirmed, try matching the newest ambiguous (probe) against representatives of other clusters
        probe = ambiguous_images[-1]
        matched_cluster = None
        matched_distance = None
        for cid, rep in representatives.items():
            if cid == current_cluster:
                continue
            v_rep, d_rep = verify_pair(rep["path"], probe)
            if v_rep:
                matched_cluster = cid
                matched_distance = d_rep
                break

        if matched_cluster is not None:
            # Add all ambiguous images to the matched existing cluster
            for a in ambiguous_images:
                clusters[matched_cluster].append(a)
                assignments[a] = {
                    "cluster_id": matched_cluster,
                    "compared_with": representatives[matched_cluster]["path"],
                    "verified": True,
                    "distance": matched_distance
                }
                # update that cluster's representative if needed
                rep = representatives[matched_cluster]
                b = calculate_blur_score(a); s = get_image_size(a)
                if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                    representatives[matched_cluster] = {"path": a, "blur": b, "size": s}
            # switch current cluster pointer to the matched cluster (so further frames compare to that cluster's last_confirmed)
            current_cluster = matched_cluster
            # set last_confirmed_img to the last ambiguous added
            last_confirmed_img = ambiguous_images[-1]
            ambiguous_images = []
            consecutive_non_matches = 0
            continue

        # nothing matched yet
        # If the number of consecutive non-matches reaches threshold -> make a new cluster from ambiguous images
        if consecutive_non_matches >= CON_FRAMES:
            current_cluster += 1
            clusters[current_cluster].extend(ambiguous_images)
            # assign meta for each
            for a in ambiguous_images:
                assignments[a] = {
                    "cluster_id": current_cluster,
                    "compared_with": last_confirmed_img,
                    "verified": False,
                    "distance": None
                }
            # compute representative for the new cluster
            representatives[current_cluster] = compute_best_representative(ambiguous_images)
            last_confirmed_img = ambiguous_images[-1]
            ambiguous_images = []
            consecutive_non_matches = 0
            continue

        # else: leave ambiguous_images unresolved and continue (they might be resolved by future frames)
        # do NOT append them to any cluster yet

    # End for loop: finalize leftover ambiguous images (stream ended)
    if ambiguous_images:
        # Try to attach to current cluster by comparing with last_confirmed
        attached = False
        for amb in ambiguous_images:
            v_last, d_last = verify_pair(last_confirmed_img, amb)
            if v_last:
                # attach all ambiguous to current cluster
                for a in ambiguous_images:
                    clusters[current_cluster].append(a)
                    assignments[a] = {
                        "cluster_id": current_cluster,
                        "compared_with": last_confirmed_img,
                        "verified": True,
                        "distance": d_last
                    }
                    rep = representatives[current_cluster]
                    b = calculate_blur_score(a); s = get_image_size(a)
                    if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                        representatives[current_cluster] = {"path": a, "blur": b, "size": s}
                attached = True
                break
        if not attached:
            # try matching to any representative
            probe = ambiguous_images[-1]
            matched_cluster = None
            matched_distance = None
            for cid, rep in representatives.items():
                if cid == current_cluster:
                    continue
                v_rep, d_rep = verify_pair(rep["path"], probe)
                if v_rep:
                    matched_cluster = cid
                    matched_distance = d_rep
                    break
            if matched_cluster is not None:
                for a in ambiguous_images:
                    clusters[matched_cluster].append(a)
                    assignments[a] = {
                        "cluster_id": matched_cluster,
                        "compared_with": representatives[matched_cluster]["path"],
                        "verified": True,
                        "distance": matched_distance
                    }
                    rep = representatives[matched_cluster]
                    b = calculate_blur_score(a); s = get_image_size(a)
                    if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                        representatives[matched_cluster] = {"path": a, "blur": b, "size": s}
            else:
                # conservative fallback: attach to current cluster
                for a in ambiguous_images:
                    clusters[current_cluster].append(a)
                    assignments[a] = {
                        "cluster_id": current_cluster,
                        "compared_with": last_confirmed_img,
                        "verified": False,
                        "distance": None
                    }
                    rep = representatives[current_cluster]
                    b = calculate_blur_score(a); s = get_image_size(a)
                    if b > rep["blur"] or (b == rep["blur"] and s > rep["size"]):
                        representatives[current_cluster] = {"path": a, "blur": b, "size": s}

        ambiguous_images = []

    return clusters, representatives, assignments


# -------------------
# Run Pipeline & Export
# -------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    image_paths = sorted(
        [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not image_paths:
        raise ValueError(f"No images found in {IMG_DIR}")

    start_time = datetime.now()
    print(f"Processing {len(image_paths)} images...")

    clusters, reps, assignments = cluster_faces(image_paths)

    # Build results in the original sequential order
    results = []
    for p in image_paths:
        assign = assignments.get(p, None)
        frame_num = extract_frame_number(p)
        if assign is None:
            # Shouldn't happen, but be robust
            assign = {"cluster_id": -1, "compared_with": None, "verified": False, "distance": None}
        rep_path = reps.get(assign["cluster_id"], {}).get("path") if assign["cluster_id"] in reps else None
        results.append({
            "frame": frame_num,
            "filename": os.path.basename(p),
            "cluster_id": assign["cluster_id"],
            "compared_with": os.path.basename(assign["compared_with"]) if assign["compared_with"] else None,
            "verified": assign["verified"],
            "distance": assign["distance"],
            "representative": os.path.basename(rep_path) if rep_path else None
        })

    print(f"\nTotal clusters found: {len(clusters)}")
    for cid, imgs in clusters.items():
        print(f" - Cluster {cid}: {len(imgs)} frames, Representative: {os.path.basename(reps[cid]['path']) if cid in reps else 'None'}")

    # Save results
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["frame", "filename", "cluster_id", "compared_with", "verified", "distance", "representative"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(OUTPUT_JSON, "w") as jf:
        json.dump(results, jf, indent=2)

    print(f"\nSaved results to {OUTPUT_CSV} and {OUTPUT_JSON}")
    print(f"Completed in {(datetime.now() - start_time).total_seconds():.2f}s")