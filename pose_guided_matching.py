"""Pose-guided face-chest matching with the shoulder-line constraint (Sec. 3.3, Eqs. 1-3).

Matching procedure for one detected chest:
  1. Each YOLOv8-pose skeleton k is summarized by the tightest box enclosing its
     reliable keypoints (keypoint confidence >= 0.50).
  2. The chest box B_C and every face box B_F are assigned to the skeleton index k
     that maximizes the Intersection-over-Area (IoA) between the box and the skeleton box.
  3. The chest is accepted only if its center satisfies the shoulder-line constraint
     of Eq. (3) for its skeleton k:
         |(c - m_s) . e_s|                  <= alpha    * ||p_rs - p_ls||
         beta_min * ||p_rs - p_ls|| <= (c - m_s) . e_n <= beta_max * ||p_rs - p_ls||
  4. Among the faces assigned to the same skeleton k, the face whose center is closest
     to the chest center (Eq. 1) is selected.

If no face can be matched (missing keypoints or faces, constraint violated), the
functions return None and the caller keeps the primary detector's decision (fallback rule).

Keypoints are expected in the YOLOv8-pose format: an array of shape (17, 3) with
(x, y, confidence) per COCO keypoint.
"""
import numpy as np

# COCO keypoint indices
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

# Hyperparameters (Sec. 3.3)
KEYPOINT_CONF_THRESHOLD = 0.50
ALPHA = 0.6       # Horizontal bound along the shoulder axis
BETA_MIN = 0.2    # Lower bound along the torso normal
BETA_MAX = 1.2    # Upper bound along the torso normal


def box_center(box):
    """Center (x, y) of an [x1, y1, x2, y2] box."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def skeleton_box(keypoints, kpt_thr=KEYPOINT_CONF_THRESHOLD):
    """Tightest [x1, y1, x2, y2] box enclosing the reliable keypoints, or None."""
    kps = np.asarray(keypoints)
    valid = kps[kps[:, 2] >= kpt_thr]
    if len(valid) == 0:
        return None
    return np.array([valid[:, 0].min(), valid[:, 1].min(), valid[:, 0].max(), valid[:, 1].max()])


def intersection_over_area(box, ref_box):
    """Intersection area of box and ref_box divided by the area of box."""
    ix1, iy1 = max(box[0], ref_box[0]), max(box[1], ref_box[1])
    ix2, iy2 = min(box[2], ref_box[2]), min(box[3], ref_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / area if area > 0 else 0.0


def assign_skeleton(box, skeleton_boxes):
    """Index k of the skeleton that maximizes IoA with box, or None if there is no overlap."""
    best_k, best_ioa = None, 0.0
    for k, sk_box in enumerate(skeleton_boxes):
        if sk_box is None:
            continue
        ioa = intersection_over_area(box, sk_box)
        if ioa > best_ioa:
            best_k, best_ioa = k, ioa
    return best_k


def shoulder_frame(keypoints, kpt_thr=KEYPOINT_CONF_THRESHOLD):
    """Shoulder-line coordinate system of Eq. (2).

    Returns (m_s, e_s, e_n, shoulder_width), or None if either shoulder is unreliable.
    e_n is the unit normal of the shoulder axis oriented downward in image coordinates
    (from the shoulders toward the lower body).
    """
    kps = np.asarray(keypoints)
    if kps[LEFT_SHOULDER, 2] < kpt_thr or kps[RIGHT_SHOULDER, 2] < kpt_thr:
        return None
    p_ls = kps[LEFT_SHOULDER, :2].astype(float)
    p_rs = kps[RIGHT_SHOULDER, :2].astype(float)
    width = np.linalg.norm(p_rs - p_ls)
    if width <= 0:
        return None

    m_s = (p_ls + p_rs) / 2.0
    e_s = (p_rs - p_ls) / width
    # Rotate e_s by 90 degrees and pick the direction pointing down the image (+y)
    e_n = np.array([-e_s[1], e_s[0]])
    if e_n[1] < 0:
        e_n = -e_n
    return m_s, e_s, e_n, width


def satisfies_shoulder_constraint(chest_center, keypoints, alpha=ALPHA,
                                  beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Check Eq. (3) for a chest center against one skeleton."""
    frame = shoulder_frame(keypoints)
    if frame is None:
        return False
    m_s, e_s, e_n, width = frame
    offset = np.asarray(chest_center, dtype=float) - m_s
    horizontal = abs(np.dot(offset, e_s))
    vertical = np.dot(offset, e_n)
    return horizontal <= alpha * width and beta_min * width <= vertical <= beta_max * width


def match_face_to_chest(chest_box, faces, all_keypoints):
    """Return (face, skeleton_index) for the face that belongs to the same person as the chest.

    chest_box: [x1, y1, x2, y2]
    faces: list of dicts with a 'box' key ([x1, y1, x2, y2]), e.g. YOLOv8l-face detections
    all_keypoints: list of (17, 3) keypoint arrays, one per person (YOLOv8-pose)

    Returns (None, None) when no face can be matched (fallback rule).
    """
    if not faces or not all_keypoints:
        return None, None

    skeleton_boxes = [skeleton_box(kps) for kps in all_keypoints]

    # Assign the chest to the skeleton with the highest IoA
    k = assign_skeleton(chest_box, skeleton_boxes)
    if k is None:
        return None, None

    # Accept the chest only if it lies in the plausible chest region of skeleton k (Eq. 3)
    chest_center = box_center(chest_box)
    if not satisfies_shoulder_constraint(chest_center, all_keypoints[k]):
        return None, None

    # Among faces assigned to the same skeleton k, pick the closest one (Eq. 1)
    best_face, best_dist = None, float('inf')
    for face in faces:
        if assign_skeleton(face['box'], skeleton_boxes) != k:
            continue
        dist = np.linalg.norm(chest_center - box_center(face['box']))
        if dist < best_dist:
            best_face, best_dist = face, dist

    if best_face is None:
        return None, None
    return best_face, k


if __name__ == "__main__":
    # Sanity check with two synthetic, side-by-side people
    def make_person(cx):
        kps = np.zeros((17, 3))
        kps[NOSE] = [cx, 100, 0.9]
        kps[LEFT_EYE] = [cx + 8, 92, 0.9]
        kps[RIGHT_EYE] = [cx - 8, 92, 0.9]
        kps[LEFT_SHOULDER] = [cx + 50, 160, 0.9]
        kps[RIGHT_SHOULDER] = [cx - 50, 160, 0.9]
        kps[11] = [cx + 30, 320, 0.9]  # Left hip
        kps[12] = [cx - 30, 320, 0.9]  # Right hip
        return kps

    people = [make_person(200), make_person(320)]
    faces = [{'box': [180, 70, 220, 120], 'id': 'A'}, {'box': [300, 70, 340, 120], 'id': 'B'}]

    # Chest of person B is matched to face B
    face, k = match_face_to_chest([290, 180, 350, 240], faces, people)
    print("chest of B ->", face['id'] if face else None, "skeleton", k)
    assert face['id'] == 'B' and k == 1

    # Box on the abdomen is rejected by the torso-normal bound (beta_max)
    face, k = match_face_to_chest([290, 300, 350, 330], faces, people)
    print("abdomen box ->", face, k)
    assert face is None

    # No pose information -> fallback
    assert match_face_to_chest([290, 180, 350, 240], faces, []) == (None, None)
    print("All checks passed.")
