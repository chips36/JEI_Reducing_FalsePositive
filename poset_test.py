import os
from ultralytics import YOLO
import cv2
import numpy as np
import time


class MultiPersonExperiment:
    """Multi-person face-chest matching experiment (Table 1).

    Compares three face assignment strategies on images with two or more people:
      - map1: detection order / confidence-based (first face returned by the detector)
      - map2: center distance-based (face closest to the chest center)
      - map3: proposed pose-guided matching (YOLOv8-pose skeleton)
    """

    def __init__(self, input_path, output_root, confidence_threshold=0.5,
                 breast_model_path='best.pt', face_model_path='yolov8l-face.pt'):
        self.input_path = input_path
        self.output_root = output_root

        # 1. Load the models
        self.breast_model = YOLO(breast_model_path)
        print(f"🚀 YOLO 가슴 모델 로드 완료: {breast_model_path}")
        self.face_model = YOLO(face_model_path)
        print(f"🚀 YOLO 얼굴 모델 로드 완료: {face_model_path}")
        # Load the pose model
        self.pose_model = YOLO('yolov8n-pose.pt')
        print("🚀 YOLO 포즈 모델 로드 완료: yolov8n-pose.pt")

        self.CONFIDENCE_THRESHOLD = confidence_threshold

        # COCO keypoint connection pairs (skeleton drawn for map3 only)
        self.SKELETON_EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face (nose-eyes-ears)
            (5, 6),  # Shoulder line
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms (shoulder-elbow-wrist)
            (5, 11), (6, 12), (11, 12),  # Torso (shoulders-hips)
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs (hip-knee-ankle)
        ]

        # Create one output folder per strategy
        self.save_folders = {
            'map1': os.path.join(output_root, 'map1_first_array'),
            'map2': os.path.join(output_root, 'map2_min_distance'),
            'map3': os.path.join(output_root, 'map3_pose_proposed')
        }
        for folder in self.save_folders.values():
            os.makedirs(folder, exist_ok=True)

    def detect_breast(self, img):
        """Run the chest detector and keep chest classes only (belly etc. are excluded)."""
        results = self.breast_model(img, conf=self.CONFIDENCE_THRESHOLD, verbose=False)[0]
        if len(results.boxes) == 0:
            return None

        valid_detections = []
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            cls_int = int(cls)

            # 0: BELLY excluded; keep only 1: BREAST_F and 2: BREAST_M
            if cls_int in [1, 2]:
                valid_detections.append({'box': box, 'class': cls_int, 'confidence': conf})

        if not valid_detections:
            return None

        # Return the single chest detection with the highest confidence
        best = max(valid_detections, key=lambda x: x['confidence'])
        return best

    def detect_faces(self, img):
        """Run the face detector and return face boxes and confidences."""
        results = self.face_model(img, verbose=False)[0]
        if len(results.boxes) == 0: return []
        faces = []
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            faces.append({'box': box, 'confidence': conf})
        return faces

    def get_pose_keypoints(self, img):
        """Run the pose model (multi-person) and return per-person keypoints (x, y, conf)."""
        results = self.pose_model(img, verbose=False)[0]
        if len(results.keypoints.data) == 0: return []
        return [kp.cpu().numpy() for kp in results.keypoints.data]

    def _get_pose_score(self, chest_center, face_center, keypoints, img_shape):
        """Score a (skeleton, face) pair for the chest with the proposed pose-guided rule.

        Higher is better; -1.0 means the skeleton has too few keypoints.
        """
        if len(keypoints) < 13: return -1.0

        # 1. Head-face score: proximity between the nose keypoint and the face center
        neck_score = -1.0
        if keypoints[0][2] > 0.5:
            neck_center = (keypoints[0][0], keypoints[0][1])
            neck_dist = np.sqrt((neck_center[0] - face_center[0]) ** 2 + (neck_center[1] - face_center[1]) ** 2)
            neck_score = 1.0 / (neck_dist + 1.0)

        # 2. Shoulder-chest score: proximity between the shoulder midpoint and the chest center
        shoulder_score = -1.0
        if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:
            shoulder_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
            shoulder_center_y = (keypoints[5][1] + keypoints[6][1]) / 2
            shoulder_center = (shoulder_center_x, shoulder_center_y)
            shoulder_dist = np.sqrt(
                (shoulder_center[0] - chest_center[0]) ** 2 + (shoulder_center[1] - chest_center[1]) ** 2)
            shoulder_score = 1.0 / (shoulder_dist + 1.0)

        base_score = neck_score + shoulder_score

        # 3. Multi-person weight: favor chests closer to the image center
        h, w = img_shape[:2]
        img_center = (w / 2, h / 2)
        dist_to_img_center = np.sqrt((chest_center[0] - img_center[0]) ** 2 + (chest_center[1] - img_center[1]) ** 2)
        center_weight = 1.0 / (dist_to_img_center / max(w, h) + 0.1)

        return base_score * center_weight

    def draw_skeleton(self, img, keypoints):
        """Draw the selected person's pose skeleton with thick lines and joints (map3 only)."""
        # 1. Draw keypoints
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.5:  # Draw only keypoints with confidence above 0.5
                cv2.circle(img, (int(x), int(y)), 7, (0, 255, 0), -1)  # Green dot

        # 2. Draw skeleton edges
        for edge in self.SKELETON_EDGES:
            pt1_idx, pt2_idx = edge
            if keypoints[pt1_idx][2] > 0.5 and keypoints[pt2_idx][2] > 0.5:
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                cv2.line(img, pt1, pt2, (255, 0, 255), 4)  # Magenta line (thickness 4)
        return img

    def compare_mapping_strategies(self, img, img_name):
        """Apply the three mapping strategies to one image and save the results
        (the pose skeleton is drawn for map3 only)."""
        processed_images = {}

        print(f"\n[시작] 🎯 {img_name} 분석 중...")

        # Stage 1: chest detection (belly fully filtered out)
        chest = self.detect_breast(img)
        print(f"  -> 1. 가슴(Breast) 검출 완료 (결과: {'성공' if chest else '없음'})")
        if chest is None: return None

        # Stage 2: face detection
        faces = self.detect_faces(img)
        print(f"  -> 2. 얼굴 검출 완료 (발견된 얼굴: {len(faces)}개)")
        if not faces: return None

        # Stage 3: pose keypoint estimation
        all_keypoints = self.get_pose_keypoints(img)
        print(f"  -> 3. 포즈 연산 완료 (발견된 사람 포즈: {len(all_keypoints)}개)")

        # Per-strategy result structure
        mapping_results = {
            'map1': {'face': faces[0]},
            'map2': {'face': None},
            'map3': {'face': None, 'kp': None}  # Only map3 keeps the matched keypoints
        }

        chest_box = chest['box']
        chest_center = ((chest_box[0] + chest_box[2]) / 2, (chest_box[1] + chest_box[3]) / 2)

        # --- [Strategy 2: map2 - minimum center distance] ---
        min_distance = float('inf')
        closest_face = None
        for face in faces:
            fb = face['box']
            face_center = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2)
            dist = np.sqrt((chest_center[0] - face_center[0]) ** 2 + (chest_center[1] - face_center[1]) ** 2)
            if dist < min_distance:
                min_distance, closest_face = dist, face

        mapping_results['map2']['face'] = closest_face

        # --- [Strategy 3: map3 - proposed pose-guided matching] ---
        best_pose_match = None
        best_kp_match = None
        if all_keypoints:
            best_score = -2.0
            # Evaluate every (skeleton, face) pair and keep the best-scoring one
            for keypoints in all_keypoints:
                for face in faces:
                    fb = face['box']
                    face_center = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2)
                    score = self._get_pose_score(chest_center, face_center, keypoints, img.shape)
                    if score > best_score:
                        best_score, best_pose_match, best_kp_match = score, face, keypoints

            mapping_results['map3']['face'] = best_pose_match
            mapping_results['map3']['kp'] = best_kp_match

        # Stage 4: draw and save the result images
        print("  -> 4. 실험 결과 개별 시각화 및 저장 중...")
        for key, result in mapping_results.items():
            face_match = result['face']

            if face_match:
                draw_img = img.copy()

                # Draw the pose skeleton for map3 only (not for map1 / map2)
                if key == 'map3' and result.get('kp') is not None:
                    draw_img = self.draw_skeleton(draw_img, result['kp'])

                # Face box (yellow)
                fx1, fy1, fx2, fy2 = map(int, face_match['box'])
                cv2.rectangle(draw_img, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                f_lbl = f"Face ({face_match['confidence']:.2f})"
                cv2.putText(draw_img, f_lbl, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Chest box (class 1 = female: red, class 2 = male: blue)
                cx1, cy1, cx2, cy2 = map(int, chest['box'])
                color = (0, 0, 255) if chest['class'] == 1 else (255, 0, 0)
                label = "Female Breast" if chest['class'] == 1 else "Male Breast"

                cv2.rectangle(draw_img, (cx1, cy1), (cx2, cy2), color, 3)
                c_txt = f"{label} | Conf: {chest['confidence']:.2f}"
                cv2.putText(draw_img, c_txt, (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Strategy label (MAP1, MAP2, MAP3)
                cv2.putText(draw_img, key.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                save_path = os.path.join(self.save_folders[key], f"{key}_{img_name}")
                cv2.imwrite(save_path, draw_img)
                processed_images[key] = True

        print(f"[완료] ✨ {img_name} 처리 완료!")
        return processed_images

    def run_experiment(self):
        """Run the experiment on all images (recursively) and print per-strategy statistics."""
        valid_extensions = ('.jpg', '.png', '.jpeg', '.webp')
        image_files = []

        print(f"🔍 '{self.input_path}' 폴더에서 이미지를 찾는 중...")

        for root, dirs, files in os.walk(self.input_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))

        total_count = len(image_files)
        print(f"📂 발견된 총 이미지 개수: {total_count}개")

        if total_count == 0:
            print(f"❌ '{self.input_path}' 폴더에 분석할 이미지가 없습니다.")
            return

        map1_success, map2_success, map3_success = 0, 0, 0
        breast_detected_count = 0
        mapping_attempt_count = 0

        start_time = time.time()
        print(f"\n🚀 YOLO 엔진 분석 시작")
        print(f"설정된 가슴 신뢰도 임계값: {self.CONFIDENCE_THRESHOLD}")
        print("-" * 60)

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"📸 현재 분석 중인 파일: {img_name}")
            img = cv2.imread(img_path)
            if img is None: continue

            try:
                processed_results = self.compare_mapping_strategies(img, img_name)
                if processed_results:
                    breast_detected_count += 1
                    mapping_attempt_count += 1
                    if processed_results.get('map1'): map1_success += 1
                    if processed_results.get('map2'): map2_success += 1
                    if processed_results.get('map3'): map3_success += 1
            except Exception as e:
                print(f"  ⚠️ 에러 발생 ({img_name}): {str(e)}")
                pass

        end_time = time.time()
        total_duration = end_time - start_time
        avg_time = total_duration / total_count if total_count > 0 else 0

        print("\n" + "=" * 60)
        print("📊 멀티 퍼슨 매핑 실험 최종 통계 리포트")
        print("=" * 60)
        print(f" ⏱️  총 소요 시간:        {total_duration:.2f} 초")
        print(f" ⚡  평균 분석 속도:      {avg_time:.2f} 초/장")
        print("-" * 60)
        print(f" 📂 총 처리 이미지:       {total_count} 개")
        print(f" ✔️ 오직 가슴(Breast)만 검출 성공: {breast_detected_count} 개")
        print(f" 🔄 매핑 시도 (가슴+얼굴): {mapping_attempt_count} 개")
        print("-" * 60)
        print(f" [매핑 전략별 성공 건수]")
        print(f"  - Map 1 (First Arr):   {map1_success} 개 / {mapping_attempt_count}")
        print(f"  - Map 2 (Min Dist):    {map2_success} 개 / {mapping_attempt_count}")
        print(f"  - Map 3 (Pose Proposed):{map3_success} 개 / {mapping_attempt_count}")
        print("=" * 60)


if __name__ == "__main__":
    # --- Settings ---
    INPUT_DIR = './ai_gen'  # Multi-person images generated with Google Gemini
    OUTPUT_DIR = './_OUT_exp_results_AI_GEN'

    BREAST_MODEL = 'best.pt'  # Fine-tuned YOLOv8 chest detector
    FACE_MODEL = 'yolov8l-face.pt'  # YOLOv8l-face detector

    CONF_THRESHOLD = 0.5  # Chest detection confidence threshold
    # ----------------

    experiment = MultiPersonExperiment(
        input_path=INPUT_DIR,
        output_root=OUTPUT_DIR,
        confidence_threshold=CONF_THRESHOLD,
        breast_model_path=BREAST_MODEL,
        face_model_path=FACE_MODEL
    )

    experiment.run_experiment()
