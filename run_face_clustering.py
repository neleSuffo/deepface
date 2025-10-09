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

CON_FRAMES = 3  # number of consecutive non-matching frames before deciding new cluster
ENFORCE_DETECTION = False

# -------------------
# Utilities
# -------------------
_verify_cache = {}

def calculate_blur_score(image_path):
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
    basename = os.path.basename(filename)
    match = re.search(r'_(\d{6})_face', basename)
    if match:
        return int(match.group(1))
    return None


def verify_pair(img1, img2):
    """Cached DeepFace.verify wrapper -> returns (verified: bool, distance: float|None)."""
    # symmetric key
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


def compute_best_representative(image_list):
    """Return best image path (sharpest then largest) from a list, or None if empty."""
    if not image_list:
        return None
    best = None
    for p in image_list:
        b = calculate_blur_score(p)
        s = get_image_size(p)
        if best is None or b > best["blur"] or (b == best["blur"] and s > best["size"]):
            best = {"path": p, "blur": b, "size": s}
    return best


# -------------------
# Clustering (deferred assignment of ambiguous frames)
# -------------------
def cluster_faces(image_paths):
    """
    Core algorithm:
      - Never finalize ambiguous frame assignments until buffer resolution.
      - When buffer length >= CON_FRAMES:
          * if last_ambiguous matches an existing representative -> assign all buffer to that cluster
          * else -> create a new cluster and assign all buffer to it
      - If a new incoming image matches the representative of current cluster,
          assign the buffer (if any) back to the current cluster + the incoming image.
    Returns:
      clusters: dict cluster_id -> [image_paths]
      representatives: dict cluster_id -> {'path','blur','size'}
      assignments: dict image_path -> {cluster_id, compared_with, verified, distance}
    """
    clusters = defaultdict(list)
    representatives = {}
    assignments = {}  # final assignment map (image -> metadata)

    # init with first image as cluster 1 immediately
    current_cluster = 1
    first = image_paths[0]
    clusters[current_cluster].append(first)
    rep = compute_best_representative([first])
    representatives[current_cluster] = rep
    assignments[first] = {
        "cluster_id": current_cluster,
        "compared_with": None,
        "verified": True,
        "distance": None
    }

    ambiguous = []  # buffer of ambiguous images (not yet assigned)
    # last confirmed representative path (always representatives[current_cluster]['path'])
    for idx in range(1, len(image_paths)):
        cur = image_paths[idx]
        rep_path = representatives[current_cluster]["path"]

        # compare incoming image to the representative of the current cluster
        verified, distance = verify_pair(rep_path, cur)

        if verified:
            # incoming matches current cluster rep -> everything in ambiguous belongs to current cluster
            if ambiguous:
                for amb in ambiguous:
                    # compute a per-amb distance to rep_path for logging
                    v, d = verify_pair(rep_path, amb)
                    clusters[current_cluster].append(amb)
                    assignments[amb] = {
                        "cluster_id": current_cluster,
                        "compared_with": rep_path,
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
                "compared_with": rep_path,
                "verified": True,
                "distance": distance
            }
            # update representative if needed
            rep_curr = representatives[current_cluster]
            b = calculate_blur_score(cur); s = get_image_size(cur)
            if b > rep_curr["blur"] or (b == rep_curr["blur"] and s > rep_curr["size"]):
                representatives[current_cluster] = {"path": cur, "blur": b, "size": s}
            continue

        # not verified -> buffer it
        ambiguous.append(cur)

        # if buffer reaches threshold -> decide
        if len(ambiguous) >= CON_FRAMES:
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

            # assign metadata for each ambiguous frame (compute individual distances to new rep for transparency)
            for amb in ambiguous:
                v, d = verify_pair(best_rep["path"], amb)
                clusters[current_cluster].append  # just ensure structure usage (we already extended)
                assignments[amb] = {
                    "cluster_id": current_cluster,
                    "compared_with": best_rep["path"],
                    "verified": bool(v),
                    "distance": d
                }
            ambiguous = []
            continue

    # End loop: if any ambiguous frames remain (stream ended before threshold),
    # tie-break: attach them to current cluster (you can change this policy)
    if ambiguous:
        rep_path = representatives[current_cluster]["path"]
        for amb in ambiguous:
            v, d = verify_pair(rep_path, amb)
            clusters[current_cluster].append(amb)
            assignments[amb] = {
                "cluster_id": current_cluster,
                "compared_with": rep_path,
                "verified": bool(v),
                "distance": d
            }
        ambiguous = []

    return clusters, representatives, assignments


# -------------------
# Run & export
# -------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image_paths = sorted(
        [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
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
            "filename": os.path.basename(p),
            "cluster_id": meta["cluster_id"] if meta else -1,
            "compared_with": os.path.basename(meta["compared_with"]) if meta and meta["compared_with"] else None,
            "verified": meta["verified"] if meta else False,
            "distance": meta["distance"] if meta else None,
            "representative": rep_path
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