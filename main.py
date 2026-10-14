import os
from ultralytics import YOLO
import cv2
from nudenet import NudeDetector
import os
import pandas as pd
import shutil
import requests
from deepface import DeepFace
import numpy as np
import time

def run_inference(model_path, source_dir, save_dir):
    """Run the YOLOv8 chest detector on every image in a folder and save annotated results."""
    # 1. Load the model
    model = YOLO(model_path)

    # 2. Create the output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. Run inference on all images in the folder
    # conf: confidence score threshold (adjust as needed)
    # save: True (automatically save annotated result images)
    # project/name: output location of the saved results
    results = model.predict(
        source=source_dir,
        conf=0.50,
        save=True,
        project=save_dir,
        name='detection_results',
        exist_ok=True
    )

    print(f"추론 완료! 결과가 다음 폴더에 저장되었습니다: {os.path.join(save_dir, 'detection_results')}")


def save_thick_pose_estimation(image_source):
    """Visualize YOLOv8-pose skeletons with thick lines (used for the paper figures)."""
    # 1. Load the pose model
    model = YOLO('yolov8n-pose.pt')

    # 2. Run inference on the image
    results = model(image_source, conf=0.5)

    for i, r in enumerate(results):
        # 3. Render the result image with custom thickness parameters
        # line_width: skeleton line thickness (int)
        # kpt_radius: keypoint marker radius (int)
        annotated_frame = r.plot(
            line_width=8,  # Much thicker than the default line width
            kpt_radius=10,  # Keypoint marker size
            labels=False,  # Hide class/confidence labels
            boxes=False  # Hide bounding boxes
        )

        # 4. Build the output file name
        if isinstance(image_source, str):
            base_name = os.path.splitext(os.path.basename(image_source))[0]
            output_path = f"{base_name}_thick_result.jpg"
        else:
            output_path = f"result_{i}.jpg"

        # 5. Save the image file
        success = cv2.imwrite(output_path, annotated_frame)

        if success:
            print(f"✅ 굵은 선으로 저장이 완료되었습니다: {output_path}")
        else:
            print(f"❌ 파일 저장에 실패했습니다.")


def run_nudenet_detection(source_dir, save_dir):
    """Run the NudeNet baseline detector on a folder and save images with colored boxes."""
    # 1. Initialize the NudeNet detector
    detector = NudeDetector()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]

    for file_name in image_files:
        image_path = os.path.join(source_dir, file_name)
        detections = detector.detect(image_path)
        image = cv2.imread(image_path)

        if image is None: continue

        for det in detections:
            label = det['class']
            score = det['score']
            box = det['box']  # [x_min, y_min, width, height]
            x, y, w, h = [int(v) for v in box]

            # --- Box color per class (BGR order) ---
            if label == 'MALE_BREAST_EXPOSED':
                color = (255, 0, 0)  # Blue
            elif label == 'FEMALE_BREAST_EXPOSED':
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green for all other classes
            # ------------------------------

            # Draw the bounding box with the selected color
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Draw the label and score
            text = f"{label}: {score:.2f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        save_path = os.path.join(save_dir, f"nude_{file_name}")
        cv2.imwrite(save_path, image)
        print(f"Saved: {save_path}")


def extract_male_breast_images(csv_path, source_img_dir, output_dir):
    """Copy images that contain the EXPOSED_BREAST_M class from the NudeNet
    Detector Auto-Generated Data annotations into a separate folder."""
    # 1. Load the annotation file
    df = pd.read_csv(csv_path, header=None,
                     names=['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'])

    # 2. Keep only the 'EXPOSED_BREAST_M' class
    male_breast_df = df[df['class_name'] == 'EXPOSED_BREAST_M']

    # 3. Remove duplicate images (one image may contain multiple annotations)
    unique_images = male_breast_df['image_path'].unique()

    # 4. Create the output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"폴더 생성 완료: {output_dir}")

    print(f"총 {len(unique_images)}개의 이미지를 복사합니다...")

    copy_count = 0
    for img_path in unique_images:
        # Extract the file name only (e.g. IMAGES/abc.jpg -> abc.jpg)
        img_filename = os.path.basename(img_path)
        src_path = os.path.join(source_img_dir, img_filename)
        dst_path = os.path.join(output_dir, img_filename)

        # Copy only if the file actually exists
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copy_count += 1
        else:
            # The source path may differ from the annotation path
            # print(f"파일을 찾을 수 없음: {src_path}")
            pass

    print(f"작업 완료: 총 {copy_count}장의 이미지가 '{output_dir}' 폴더에 저장되었습니다.")


# 1. Unsplash download settings (the Access Key is read from the UNSPLASH_ACCESS_KEY environment variable)
ACCESS_KEY = os.environ["UNSPLASH_ACCESS_KEY"]
QUERY = 'bodybuilder shirtless portrait'
SAVE_FOLDER = 'bodybuilding_dataset'
TOTAL_IMAGES = 1000
PER_PAGE = 30  # Number of images per page (max 30)

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def download_images():
    """Collect male-chest images through the Unsplash developer API."""
    downloaded_count = 0
    page = 1

    print(f"데이터 수집 시작: {QUERY}...")

    while downloaded_count < TOTAL_IMAGES:
        # Call the Unsplash search API
        url = f"https://api.unsplash.com/search/photos?query={QUERY}&client_id={ACCESS_KEY}&per_page={PER_PAGE}&page={page}"
        response = requests.get(url)

        if response.status_code != 200:
            print("API 호출 오류:", response.text)
            break

        data = response.json()
        photos = data.get('results', [])

        if not photos:
            print("더 이상 검색 결과가 없습니다.")
            break

        for photo in photos:
            if downloaded_count >= TOTAL_IMAGES:
                break

            # Use the high-resolution image URL ('regular' or 'full')
            img_url = photo['urls']['regular']
            img_id = photo['id']

            try:
                img_data = requests.get(img_url).content
                with open(f"{SAVE_FOLDER}/{img_id}.jpg", 'wb') as f:
                    f.write(img_data)
                downloaded_count += 1
                print(f"[{downloaded_count}/{TOTAL_IMAGES}] 다운로드 완료: {img_id}.jpg")
            except Exception as e:
                print(f"다운로드 실패 ({img_id}): {e}")

        page += 1

    print(f"\n총 {downloaded_count}장의 이미지가 '{SAVE_FOLDER}' 폴더에 저장되었습니다.")


def filter_adult_content_images(source_folder, output_folder, face_model_path, breast_model_path):
    """
    Keep only images in which at least one face and a male/female chest are both detected,
    and copy them to the output folder.
    """
    # 1. Create the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Load the models
    face_model = YOLO(face_model_path)
    breast_model = YOLO(breast_model_path)

    # Target class indices from the dataset yaml (1: EXPOSED_BREAST_F, 2: EXPOSED_BREAST_M)
    target_classes = [1, 2]

    # 3. Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

    print(f"총 {len(image_files)}개의 이미지를 분석합니다...")

    count = 0
    for file_name in image_files:
        file_path = os.path.join(source_folder, file_name)

        # Run face detection
        face_results = face_model(file_path, verbose=False)

        # Continue only if at least one face is detected
        if len(face_results[0].boxes) >= 1:
            # Run chest detection
            breast_results = breast_model(file_path, verbose=False)
            detected_classes = breast_results[0].boxes.cls.tolist()

            # Check whether class 1 (female chest) or 2 (male chest) is present
            if any(cls in target_classes for cls in detected_classes):
                shutil.copy(file_path, os.path.join(output_folder, file_name))
                count += 1
                print(f"[{count}] 조건 충족 파일 복사 완료: {file_name}")

    print(f"작업 완료. 총 {count}개의 파일이 '{output_folder}'에 저장되었습니다.")


#####################################################################################################################################\

class BodybuildingExperiment:
    """Static-image experiment for the proposed multi-stage verification pipeline:
    chest detection -> pose estimation -> face detection -> DeepFace gender verification.
    Images are sorted into man/woman/unknown folders according to the final decision."""

    def __init__(self, input_path, output_path, confidence_threshold=0.90,
                 detector_type='yolo',  # Primary chest detector: 'yolo' or 'nudenet'
                 breast_model_path='best.pt', face_model_path='yolov8l-face.pt'):
        self.input_path = input_path
        self.output_path = output_path
        self.detector_type = detector_type.lower()

        # 1. Load the models
        if self.detector_type == 'yolo':
            self.breast_model = YOLO(breast_model_path)
            print(f"🚀 YOLO 엔진 로드 완료: {breast_model_path}")
        else:
            self.nude_detector = NudeDetector()
            print("🚀 NudeNet 엔진 로드 완료")

        self.face_model = YOLO(face_model_path)
        self.pose_model = YOLO('yolov8n-pose.pt')

        # Class IDs of the fine-tuned YOLOv8 chest detector
        self.BELLY = 0
        self.BREAST_F = 1
        self.BREAST_M = 2
        self.BUTTOCKS = 3
        self.GENITALIA_F = 4
        self.GENITALIA_M = 5

        # NudeNet label -> internal class ID mapping (used to interpret NudeNet outputs)
        self.NUDENET_MAP = {
            'FEMALE_BREAST_EXPOSED': self.BREAST_F,
            'FEMALE_BREAST_COVERED': self.BREAST_F,
            'MALE_BREAST_EXPOSED': self.BREAST_M,
            'BUTTOCKS_EXPOSED': self.BUTTOCKS,
            'FEMALE_GENITALIA_EXPOSED': self.GENITALIA_F,
            'MALE_GENITALIA_EXPOSED': self.GENITALIA_M
        }

        # Gender confidence threshold (tau) required to override the detector decision
        self.DEEPFACE_CONFIDENCE_THRESHOLD = confidence_threshold

        # Create the output folders
        self.folders = {
            'Man': os.path.join(output_path, 'man'),
            'Woman': os.path.join(output_path, 'woman'),
            'Unknown': os.path.join(output_path, 'unknown')
        }
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def breast_class_to_gender(self, breast_class):
        """Map a chest class ID to the gender implied by the primary detector."""
        if breast_class == self.BREAST_F:
            return 'Woman'
        elif breast_class == self.BREAST_M:
            return 'Man'
        return 'Unknown'

    def _detect_with_yolo(self, img):
        """Chest detection with the fine-tuned YOLOv8 model."""
        results = self.breast_model(img, verbose=False)[0]
        if len(results.boxes) == 0:
            return None

        detections = []
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            cls_int = int(cls)
            # Keep chest classes only
            if cls_int in [self.BREAST_F, self.BREAST_M]:
                detections.append({'box': box, 'confidence': conf, 'class': cls_int})
        return detections

    def _detect_with_nudenet(self, img_path):
        """Chest detection with NudeNet (robust to output format differences between versions)."""
        try:
            # Recent NudeNet versions return a list of detections
            raw_results = self.nude_detector.detect(img_path)

            detections = []
            for res in raw_results:
                # Depending on the version, the key may be 'label' or 'class'
                label = res.get('label') or res.get('class')
                score = res.get('score') or res.get('confidence', 0.0)
                box = res.get('box')

                if label and label in self.NUDENET_MAP:
                    mapped_cls = self.NUDENET_MAP[label]
                    # Keep chest classes (1, 2) only
                    if mapped_cls in [self.BREAST_F, self.BREAST_M]:
                        detections.append({
                            'box': box,  # [x1, y1, x2, y2]
                            'confidence': score,
                            'class': mapped_cls
                        })
            return detections
        except Exception as e:
            print(f"  ⚠️  NudeNet 검출 중 상세 에러 발생: {str(e)}")
            return []

    def detect_breast(self, img, img_path):
        """Unified chest detection interface for the selected primary detector."""
        if self.detector_type == 'yolo':
            detections = self._detect_with_yolo(img)
        else:
            detections = self._detect_with_nudenet(img_path)

        if not detections:
            return None, None

        # Return the detection with the highest confidence
        best = max(detections, key=lambda x: x['confidence'])
        return best['box'], best['class']

    def detect_faces(self, img):
        """Detect faces with YOLOv8l-face and return their boxes and confidences."""
        results = self.face_model(img, verbose=False)[0]
        if len(results.boxes) == 0: return []
        faces = []
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            faces.append({'box': box, 'confidence': conf})
        return faces

    def get_pose_keypoints(self, img):
        """Estimate the 17 COCO keypoints (x, y, conf) of every person with YOLOv8-pose."""
        results = self.pose_model(img, verbose=False)[0]
        if len(results.keypoints.data) == 0: return []
        return [kp.cpu().numpy() for kp in results.keypoints.data]

    def match_face_to_chest(self, chest_box, faces, all_keypoints):
        """Return the face that belongs to the same person as the detected chest.

        A single face is matched directly. Otherwise, for each skeleton with reliable
        shoulders (keypoints 5, 6) and nose (keypoint 0), the face closest to the nose is
        considered, and the skeleton whose shoulder midpoint is closest to the chest wins.
        Returns None when no face can be matched (fallback to the detector decision).
        """
        if len(faces) == 1: return faces[0]
        if not all_keypoints: return None
        chest_center_x = (chest_box[0] + chest_box[2]) / 2
        chest_center_y = (chest_box[1] + chest_box[3]) / 2
        best_match, min_distance = None, float('inf')
        for keypoints in all_keypoints:
            if len(keypoints) < 13: continue
            # Both shoulders must exceed the keypoint confidence threshold (0.5)
            if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:
                shoulder_x = (keypoints[5][0] + keypoints[6][0]) / 2
                shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
                # Distance between the shoulder midpoint and the chest center
                dist = np.sqrt((shoulder_x - chest_center_x) ** 2 + (shoulder_y - chest_center_y) ** 2)
                if keypoints[0][2] > 0.5:
                    nx, ny = keypoints[0][0], keypoints[0][1]
                    for face in faces:
                        fb = face['box']
                        fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                        # Face must lie near the nose keypoint of the same skeleton
                        f_dist = np.sqrt((nx - fcx) ** 2 + (ny - fcy) ** 2)
                        if f_dist < 100 and dist < min_distance:
                            min_distance, best_match = dist, face
        return best_match

    def classify_gender_from_face(self, img, face_box):
        """Classify the perceived gender of a face crop with DeepFace.

        The crop is passed to DeepFace as an in-memory BGR array and is never written to disk.
        Returns (dominant_gender, confidence in [0, 1]) or ('Unknown', 0.0) on failure.
        """
        try:
            x1, y1, x2, y2 = map(int, face_box)
            face_img = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if face_img.size == 0: return 'Unknown', 0.0
            analysis = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False, silent=True)
            res = analysis[0]
            return res['dominant_gender'], res['gender'][res['dominant_gender']] / 100.0
        except:
            return 'Unknown', 0.0

    def draw_results(self, img, gender, face_box, chest_box, breast_class, source='breast', confidence=None):
        """Draw the matched face box (yellow) and the chest box (red: female, blue: male)."""
        if face_box is not None:
            x1, y1, x2, y2 = map(int, face_box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            lbl = f"Face ({confidence:.2f})" if confidence else "Face"
            cv2.putText(img, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        color = (0, 0, 255) if breast_class == self.BREAST_F else (255, 0, 0)
        label = "Female Breast" if breast_class == self.BREAST_F else "Male Breast"
        x1, y1, x2, y2 = map(int, chest_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        src_txt = f"{label} ({gender}) | {source}"
        cv2.putText(img, src_txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def run_experiment(self):
        """Run the full pipeline on all images and print detailed statistics (including timing)."""
        image_files = [f for f in os.listdir(self.input_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]

        total_count = len(image_files)
        if total_count == 0:
            print("❌ 분석할 이미지가 없습니다.")
            return

        # --- Statistics counters ---
        model_man, model_woman = 0, 0
        final_man, final_woman = 0, 0
        unknown_count = 0

        # --- Start timing ---
        start_time = time.time()

        print(f"\n🚀 {self.detector_type.upper()} 엔진 분석 시작")
        print(f"설정된 Face 신뢰도 임계값: {self.DEEPFACE_CONFIDENCE_THRESHOLD}")
        print("-" * 60)

        for img_name in image_files:
            img_path = os.path.join(self.input_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                unknown_count += 1
                continue

            try:
                # Stage 1: chest detection
                chest_box, breast_class = self.detect_breast(img, img_path)

                if chest_box is None:
                    unknown_count += 1
                    cv2.imwrite(os.path.join(self.folders['Unknown'], img_name), img)
                    continue

                # Count the primary detector decision (baseline)
                gender_from_model = self.breast_class_to_gender(breast_class)
                if gender_from_model == 'Man':
                    model_man += 1
                elif gender_from_model == 'Woman':
                    model_woman += 1

                # Stage 2: pose-guided face matching and gender classification
                faces = self.detect_faces(img)
                matched_face = self.match_face_to_chest(chest_box, faces,
                                                        self.get_pose_keypoints(img)) if faces else None

                # Fallback rule: keep the detector decision unless overridden below
                final_gender = gender_from_model
                gender_source = 'breast'
                face_box, face_conf = None, None

                if matched_face:
                    g_face, conf = self.classify_gender_from_face(img, matched_face['box'])
                    face_box, face_conf = matched_face['box'], conf
                    # Override the detector decision only when the gender confidence >= tau
                    if conf >= self.DEEPFACE_CONFIDENCE_THRESHOLD:
                        final_gender = g_face
                        gender_source = 'face'

                # Count the final decision
                if final_gender == 'Man':
                    final_man += 1
                elif final_gender == 'Woman':
                    final_woman += 1
                else:
                    unknown_count += 1

                # Save the annotated result
                processed = self.draw_results(img.copy(), final_gender, face_box, chest_box, breast_class,
                                              gender_source, face_conf)
                cv2.imwrite(os.path.join(self.folders[final_gender], img_name), processed)

            except Exception as e:
                unknown_count += 1
                cv2.imwrite(os.path.join(self.folders['Unknown'], img_name), img)

        # --- Stop timing ---
        end_time = time.time()
        total_duration = end_time - start_time
        avg_time = total_duration / total_count if total_count > 0 else 0

        # ==========================================
        # Final statistics report
        # ==========================================
        print("\n" + "=" * 60)
        print("📊 보디빌딩 실험 최종 통계 리포트")
        print("=" * 60)
        print(f" ⚙️  Confidence Threshold: {self.DEEPFACE_CONFIDENCE_THRESHOLD:>7}")
        print(f" ⏱️  총 소요 시간:        {total_duration:>7.2f} 초")
        print(f" ⚡  평균 분석 속도:      {avg_time:>7.2f} 초/장")
        print("-" * 60)
        print(f" 📂 총 처리 이미지:       {total_count:>7} 개")
        print("-" * 60)

        print(f" [남성(Man)]")
        print(f"  - 모델에서 검출한 남자:      {model_man:>5} 개")
        print(f"  - Face까지 반영된 최종 남자: {final_man:>5} 개")
        print("-" * 60)

        print(f" [여성(Woman)]")
        print(f"  - 모델에서 검출한 여자:      {model_woman:>5} 개")
        print(f"  - Face까지 반영된 최종 여자: {final_woman:>5} 개")
        print("-" * 60)

        print(f" [기타]")
        print(f"  - 인식하지 못한 개수:        {unknown_count:>5} 개")
        print("=" * 60)



# --- Settings ---
ANNOTATIONS_CSV = 'annotations.csv'  # Annotation file path
SOURCE_IMAGES = './images'  # Folder containing all source images
TARGET_DIR = './only_male_breast'  # Output folder for male-chest images only
# ----------------


if __name__ == "__main__":
    # --- Settings ---
    MODEL_PATH = 'best.pt'  # Trained chest detector weights
    SOURCE_IMAGE_DIR = './images'  # Source image folder for detection

    OUTPUT_DIR = './runs/nude_predict'  # Root folder for result images

    TRAIN_SOURCE_IMAGE_DIR = './male_breast'  # Source image folder for detection
    AI_SOURCE_IMAGE_DIR = './AI'  # Generative-AI image folder
    TRAIN_OUTPUT_DIR = './runs/train_predict'  # Root folder for result images

    # ----------------

    NUDE_DIR = './runs/predict'  # Root folder for NudeNet result images

    # --- NudeNet Detector Auto-Generated Data paths ---
    ANNOTATIONS_CSV = 'C:/Users/kyj/Downloads/DETECTOR_AUTO_GENERATED_DATA/annotations.csv'  # Annotation file path
    SOURCE_IMAGES = 'C:/Users/kyj/Downloads/DETECTOR_AUTO_GENERATED_DATA/images'  # Folder containing all source images
    TARGET_DIR = 'C:/Users/kyj/Downloads/DETECTOR_AUTO_GENERATED_DATA/only_male_breast'  # Output folder for male-chest images only
    # ----------------

    # YOLOv8 baseline detector
    # run_inference(MODEL_PATH, AI_SOURCE_IMAGE_DIR, OUTPUT_DIR)
    # run_inference(MODEL_PATH, TRAIN_SOURCE_IMAGE_DIR, TRAIN_OUTPUT_DIR)

    # NudeNet baseline detector
    # run_nudenet_detection(AI_SOURCE_IMAGE_DIR, NUDE_DIR)

    # Extract male-chest images only
    # extract_male_breast_images(ANNOTATIONS_CSV, SOURCE_IMAGES, TARGET_DIR)

    # Download images from Unsplash
    # download_images()

    # Keep only usable images (face + chest detected) from the downloaded set
    # filter_adult_content_images(
    #    source_folder='Z:/99.KCI음란물_학습데이터셋/YOLOv8 학습한 데이터셋/test/images',
    #    output_folder='./filtered_results_NUDENET_TEST',
    #    face_model_path='yolov8l-face.pt',  # Face detector weights
    #    breast_model_path='best.pt'  # Trained chest detector weights
    # )

    # 300 images are sampled per class.
    # For each image in the folder: 1. chest detection (red: female, blue: male) 2. face detection

    #INPUT_DIR = 'filtered_results_NUDENET_TEST'  # Female dataset
    INPUT_DIR = 'MALE_DATASETS_300'  # Male dataset
    OUTPUT_DIR = '_OUT_results_FILTERED'  # Output folder


    # Create the experiment and run it
    experiment = BodybuildingExperiment(
        input_path=INPUT_DIR,
        output_path=OUTPUT_DIR,
        confidence_threshold=0.9,  # Operating threshold tau selected on the validation set
        detector_type='yolo',  # 'yolo' uses the fine-tuned YOLOv8 model, 'nudenet' uses NudeNet
        breast_model_path='best.pt'
    )


    experiment.run_experiment()

    #save_thick_pose_estimation('그림1.png')
