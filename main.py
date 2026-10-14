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
    # 1. 모델 로드
    model = YOLO(model_path)

    # 2. 결과 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. 폴더 내 이미지들에 대해 추론 수행
    # conf: 0.25 (신뢰도 임계값, 필요에 따라 조절 가능)
    # save: True (결과 이미지를 자동으로 저장)
    # project/name: 저장될 경로 설정
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
    # 1. 모델 로드
    model = YOLO('yolov8n-pose.pt')

    # 2. 이미지 추론
    results = model(image_source, conf=0.5)

    for i, r in enumerate(results):
        # 3. 결과 이미지 생성 (굵기 조절 파라미터 추가)
        # line_width: 선의 굵기 (정수)
        # kpt_radius: 관절 포인트(점)의 크기 (정수)
        annotated_frame = r.plot(
            line_width=8,  # 선의 굵기를 8로 설정 (기존보다 훨씬 굵음)
            kpt_radius=10,  # 관절 점의 크기를 10으로 설정
            labels=False,  # 클래스/정확도 라벨 표시 여부
            boxes=False  # 바운딩 박스 표시 여부
        )

        # 4. 출력 파일명 설정
        if isinstance(image_source, str):
            base_name = os.path.splitext(os.path.basename(image_source))[0]
            output_path = f"{base_name}_thick_result.jpg"
        else:
            output_path = f"result_{i}.jpg"

        # 5. 이미지 파일 저장
        success = cv2.imwrite(output_path, annotated_frame)

        if success:
            print(f"✅ 굵은 선으로 저장이 완료되었습니다: {output_path}")
        else:
            print(f"❌ 파일 저장에 실패했습니다.")


def run_nudenet_detection(source_dir, save_dir):
    # 1. NudeNet Detector 초기화
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

            # --- 색상 조건 설정 (BGR 순서) ---
            if label == 'MALE_BREAST_EXPOSED':
                color = (255, 0, 0)  # 파란색 (Blue)
            elif label == 'FEMALE_BREAST_EXPOSED':
                color = (0, 0, 255)  # 빨간색 (Red)
            else:
                color = (0, 255, 0)  # 나머지 초록색 (Green)
            # ------------------------------

            # 바운딩 박스 그리기 (설정된 color 적용)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # 라벨 및 점수 표시
            text = f"{label}: {score:.2f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        save_path = os.path.join(save_dir, f"nude_{file_name}")
        cv2.imwrite(save_path, image)
        print(f"Saved: {save_path}")


def extract_male_breast_images(csv_path, source_img_dir, output_dir):
    # 1. 어노테이션 파일 로드
    df = pd.read_csv(csv_path, header=None,
                     names=['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'])

    # 2. 'EXPOSED_BREAST_M' 클래스만 필터링
    male_breast_df = df[df['class_name'] == 'EXPOSED_BREAST_M']

    # 3. 중복 이미지 제거 (한 이미지에 여러 번 탐지된 경우 대비)
    unique_images = male_breast_df['image_path'].unique()

    # 4. 출력 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"폴더 생성 완료: {output_dir}")

    print(f"총 {len(unique_images)}개의 이미지를 복사합니다...")

    copy_count = 0
    for img_path in unique_images:
        # 파일명만 추출 (예: IMAGES/abc.jpg -> abc.jpg)
        img_filename = os.path.basename(img_path)
        src_path = os.path.join(source_img_dir, img_filename)
        dst_path = os.path.join(output_dir, img_filename)

        # 파일이 실제로 존재할 때만 복사
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copy_count += 1
        else:
            # 원본 경로가 다를 경우를 대비해 알림
            # print(f"파일을 찾을 수 없음: {src_path}")
            pass

    print(f"작업 완료: 총 {copy_count}장의 이미지가 '{output_dir}' 폴더에 저장되었습니다.")


# 1. 설정 (Access Key는 환경변수 UNSPLASH_ACCESS_KEY에서 읽음)
ACCESS_KEY = os.environ["UNSPLASH_ACCESS_KEY"]
QUERY = 'bodybuilder shirtless portrait'
SAVE_FOLDER = 'bodybuilding_dataset'
TOTAL_IMAGES = 1000
PER_PAGE = 30  # 한 페이지당 가져올 이미지 수 (최대 30)

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def download_images():
    downloaded_count = 0
    page = 1

    print(f"데이터 수집 시작: {QUERY}...")

    while downloaded_count < TOTAL_IMAGES:
        # Unsplash API 호출
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

            # 고화질 이미지 URL 선택 (regular 또는 full)
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
    얼굴이 1개 이상 검출되고, 동시에 남성/여성 가슴이 검출되는 이미지를 필터링하여 저장합니다.
    """
    # 1. 결과 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 모델 로드
    face_model = YOLO(face_model_path)
    breast_model = YOLO(breast_model_path)

    # 제공된 yaml 기준 타겟 클래스 인덱스 (1: EXPOSED_BREAST_F, 2: EXPOSED_BREAST_M)
    target_classes = [1, 2]

    # 3. 대상 이미지 확장자 설정
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

    print(f"총 {len(image_files)}개의 이미지를 분석합니다...")

    count = 0
    for file_name in image_files:
        file_path = os.path.join(source_folder, file_name)

        # 얼굴 검출 수행
        face_results = face_model(file_path, verbose=False)

        # 얼굴이 1개 이상인 경우에만 다음 단계 진행
        if len(face_results[0].boxes) >= 1:
            # 가슴 검출 수행
            breast_results = breast_model(file_path, verbose=False)
            detected_classes = breast_results[0].boxes.cls.tolist()

            # 클래스 1(여성 가슴) 또는 2(남성 가슴)가 포함되어 있는지 확인
            if any(cls in target_classes for cls in detected_classes):
                shutil.copy(file_path, os.path.join(output_folder, file_name))
                count += 1
                print(f"[{count}] 조건 충족 파일 복사 완료: {file_name}")

    print(f"작업 완료. 총 {count}개의 파일이 '{output_folder}'에 저장되었습니다.")


#####################################################################################################################################\

class BodybuildingExperiment:
    def __init__(self, input_path, output_path, confidence_threshold=0.90,
                 detector_type='yolo',  # 'yolo' 또는 'nudenet' 선택 가능
                 breast_model_path='best.pt', face_model_path='yolov8l-face.pt'):
        self.input_path = input_path
        self.output_path = output_path
        self.detector_type = detector_type.lower()

        # 1. 모델 로드
        if self.detector_type == 'yolo':
            self.breast_model = YOLO(breast_model_path)
            print(f"🚀 YOLO 엔진 로드 완료: {breast_model_path}")
        else:
            self.nude_detector = NudeDetector()
            print("🚀 NudeNet 엔진 로드 완료")

        self.face_model = YOLO(face_model_path)
        self.pose_model = YOLO('yolov8n-pose.pt')

        # 클래스 정의 (기존 체계 유지)
        self.BELLY = 0
        self.BREAST_F = 1
        self.BREAST_M = 2
        self.BUTTOCKS = 3
        self.GENITALIA_F = 4
        self.GENITALIA_M = 5

        # NudeNet 라벨 -> 내부 클래스 ID 매핑 (NudeNet 결과 해석용)
        self.NUDENET_MAP = {
            'FEMALE_BREAST_EXPOSED': self.BREAST_F,
            'FEMALE_BREAST_COVERED': self.BREAST_F,
            'MALE_BREAST_EXPOSED': self.BREAST_M,
            'BUTTOCKS_EXPOSED': self.BUTTOCKS,
            'FEMALE_GENITALIA_EXPOSED': self.GENITALIA_F,
            'MALE_GENITALIA_EXPOSED': self.GENITALIA_M
        }

        self.DEEPFACE_CONFIDENCE_THRESHOLD = confidence_threshold

        # 폴더 초기화
        self.folders = {
            'Man': os.path.join(output_path, 'man'),
            'Woman': os.path.join(output_path, 'woman'),
            'Unknown': os.path.join(output_path, 'unknown')
        }
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def breast_class_to_gender(self, breast_class):
        if breast_class == self.BREAST_F:
            return 'Woman'
        elif breast_class == self.BREAST_M:
            return 'Man'
        return 'Unknown'

    def _detect_with_yolo(self, img):
        """기존 YOLO 기반 검출 로직"""
        results = self.breast_model(img, verbose=False)[0]
        if len(results.boxes) == 0:
            return None

        detections = []
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            cls_int = int(cls)
            if cls_int in [self.BREAST_F, self.BREAST_M]:
                detections.append({'box': box, 'confidence': conf, 'class': cls_int})
        return detections

    def _detect_with_nudenet(self, img_path):
        """NudeNet 기반 검출 로직 (에러 방지 버전)"""
        try:
            # NudeNet의 최신 버전은 결과가 리스트 형태로 반환됩니다.
            raw_results = self.nude_detector.detect(img_path)

            detections = []
            for res in raw_results:
                # 버전마다 'label' 또는 'class' 키를 사용할 수 있으므로 체크
                label = res.get('label') or res.get('class')
                score = res.get('score') or res.get('confidence', 0.0)
                box = res.get('box')

                if label and label in self.NUDENET_MAP:
                    mapped_cls = self.NUDENET_MAP[label]
                    # 가슴 클래스(1, 2)만 필터링
                    if mapped_cls in [self.BREAST_F, self.BREAST_M]:
                        detections.append({
                            'box': box,  # [x1, y1, x2, y2] 형태
                            'confidence': score,
                            'class': mapped_cls
                        })
            return detections
        except Exception as e:
            print(f"  ⚠️  NudeNet 검출 중 상세 에러 발생: {str(e)}")
            return []

    def detect_breast(self, img, img_path):
        """엔진 선택에 따른 가슴 검출 통합 인터페이스"""
        if self.detector_type == 'yolo':
            detections = self._detect_with_yolo(img)
        else:
            detections = self._detect_with_nudenet(img_path)

        if not detections:
            return None, None

        # 가장 신뢰도가 높은 결과 반환
        best = max(detections, key=lambda x: x['confidence'])
        return best['box'], best['class']

    # ... [detect_faces, get_pose_keypoints, match_face_to_chest 등 기존 메서드는 동일] ...

    def detect_faces(self, img):
        results = self.face_model(img, verbose=False)[0]
        if len(results.boxes) == 0: return []
        faces = []
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            faces.append({'box': box, 'confidence': conf})
        return faces

    def get_pose_keypoints(self, img):
        results = self.pose_model(img, verbose=False)[0]
        if len(results.keypoints.data) == 0: return []
        return [kp.cpu().numpy() for kp in results.keypoints.data]

    def match_face_to_chest(self, chest_box, faces, all_keypoints):
        if len(faces) == 1: return faces[0]
        if not all_keypoints: return None
        chest_center_x = (chest_box[0] + chest_box[2]) / 2
        chest_center_y = (chest_box[1] + chest_box[3]) / 2
        best_match, min_distance = None, float('inf')
        for keypoints in all_keypoints:
            if len(keypoints) < 13: continue
            if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:
                shoulder_x = (keypoints[5][0] + keypoints[6][0]) / 2
                shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
                dist = np.sqrt((shoulder_x - chest_center_x) ** 2 + (shoulder_y - chest_center_y) ** 2)
                if keypoints[0][2] > 0.5:
                    nx, ny = keypoints[0][0], keypoints[0][1]
                    for face in faces:
                        fb = face['box']
                        fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                        f_dist = np.sqrt((nx - fcx) ** 2 + (ny - fcy) ** 2)
                        if f_dist < 100 and dist < min_distance:
                            min_distance, best_match = dist, face
        return best_match

    def classify_gender_from_face(self, img, face_box):
        try:
            x1, y1, x2, y2 = map(int, face_box)
            face_img = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if face_img.size == 0: return 'Unknown', 0.0
            temp_path = f'temp_face_{np.random.randint(1000)}.jpg'
            cv2.imwrite(temp_path, face_img)
            analysis = DeepFace.analyze(temp_path, actions=['gender'], enforce_detection=False, silent=True)
            if os.path.exists(temp_path): os.remove(temp_path)
            res = analysis[0]
            return res['dominant_gender'], res['gender'][res['dominant_gender']] / 100.0
        except:
            return 'Unknown', 0.0

    def draw_results(self, img, gender, face_box, chest_box, breast_class, source='breast', confidence=None):
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
        """전체 파이프라인 실행 및 상세 비교 통계 출력 (시간 측정 포함)"""
        image_files = [f for f in os.listdir(self.input_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]

        total_count = len(image_files)
        if total_count == 0:
            print("❌ 분석할 이미지가 없습니다.")
            return

        # --- 통계 카운터 ---
        model_man, model_woman = 0, 0
        final_man, final_woman = 0, 0
        unknown_count = 0

        # --- 시간 측정 시작 ---
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
                # 1단계: 가슴 검출
                chest_box, breast_class = self.detect_breast(img, img_path)

                if chest_box is None:
                    unknown_count += 1
                    cv2.imwrite(os.path.join(self.folders['Unknown'], img_name), img)
                    continue

                # 모델 검출 결과 카운트
                gender_from_model = self.breast_class_to_gender(breast_class)
                if gender_from_model == 'Man':
                    model_man += 1
                elif gender_from_model == 'Woman':
                    model_woman += 1

                # 2단계: 얼굴 및 성별 분석
                faces = self.detect_faces(img)
                matched_face = self.match_face_to_chest(chest_box, faces,
                                                        self.get_pose_keypoints(img)) if faces else None

                final_gender = gender_from_model
                gender_source = 'breast'
                face_box, face_conf = None, None

                if matched_face:
                    g_face, conf = self.classify_gender_from_face(img, matched_face['box'])
                    face_box, face_conf = matched_face['box'], conf
                    if conf >= self.DEEPFACE_CONFIDENCE_THRESHOLD:
                        final_gender = g_face
                        gender_source = 'face'

                # 최종 결과 카운트
                if final_gender == 'Man':
                    final_man += 1
                elif final_gender == 'Woman':
                    final_woman += 1
                else:
                    unknown_count += 1

                # 결과 저장
                processed = self.draw_results(img.copy(), final_gender, face_box, chest_box, breast_class,
                                              gender_source, face_conf)
                cv2.imwrite(os.path.join(self.folders[final_gender], img_name), processed)

            except Exception as e:
                unknown_count += 1
                cv2.imwrite(os.path.join(self.folders['Unknown'], img_name), img)

        # --- 시간 측정 종료 ---
        end_time = time.time()
        total_duration = end_time - start_time
        avg_time = total_duration / total_count if total_count > 0 else 0

        # ==========================================
        # 최종 통계 리포트 출력
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



# --- 설정 영역 ---
ANNOTATIONS_CSV = 'annotations.csv'  # 어노테이션 파일 경로
SOURCE_IMAGES = './images'  # 전체 원본 이미지가 들어있는 폴더
TARGET_DIR = './only_male_breast'  # 남성 가슴 이미지만 저장할 폴더
# ----------------


if __name__ == "__main__":
    # --- 설정 영역 ---
    MODEL_PATH = 'best.pt'  # 학습된 모델 파일 경로
    SOURCE_IMAGE_DIR = './images'  # 검출을 진행할 원본 이미지 폴더 경로

    OUTPUT_DIR = './runs/nude_predict'  # 결과 이미지가 저장될 루트 폴더

    TRAIN_SOURCE_IMAGE_DIR = './male_breast'  # 검출을 진행할 원본 이미지 폴더 경로
    AI_SOURCE_IMAGE_DIR = './AI'  # 검출을 진행할 원본 이미지 폴더 경로
    TRAIN_OUTPUT_DIR = './runs/train_predict'  # 결과 이미지가 저장될 루트 폴더

    # ----------------D

    NUDE_DIR = './runs/predict'  # 결과 이미지가 저장될 루트 폴더

    # --- 설정 영역 ---
    ANNOTATIONS_CSV = 'C:/Users/kyj/Downloads/DETECTOR_AUTO_GENERATED_DATA/annotations.csv'  # 어노테이션 파일 경로
    SOURCE_IMAGES = 'C:/Users/kyj/Downloads/DETECTOR_AUTO_GENERATED_DATA/images'  # 전체 원본 이미지가 들어있는 폴더
    TARGET_DIR = 'C:/Users/kyj/Downloads/DETECTOR_AUTO_GENERATED_DATA/only_male_breast'  # 남성 가슴 이미지만 저장할 폴더
    # ----------------

    # YOLO 학습모델
    # run_inference(MODEL_PATH, AI_SOURCE_IMAGE_DIR, OUTPUT_DIR)
    # run_inference(MODEL_PATH, TRAIN_SOURCE_IMAGE_DIR, TRAIN_OUTPUT_DIR)

    # nudenet model
    # run_nudenet_detection(AI_SOURCE_IMAGE_DIR, NUDE_DIR)

    # 남자가슴만 따로 뽑음
    # extract_male_breast_images(ANNOTATIONS_CSV, SOURCE_IMAGES, TARGET_DIR)

    # 이미지 다운로드
    # download_images()

    # 다운로드 받은 이미지중 쓸만한거 필터링
    # filter_adult_content_images(
    #    source_folder='Z:/99.KCI음란물_학습데이터셋/YOLOv8 학습한 데이터셋/test/images',
    #    output_folder='./filtered_results_NUDENET_TEST',
    #    face_model_path='yolov8l-face.pt',  # 얼굴 검출 모델 파일명
    #    breast_model_path='best.pt'  # 학습된 가슴 검출 모델 파일명
    # )

    # 이미지 300 장씩 추출함.
    # 이 폴더에 있는 이미지를가지고, 1. 가슴검출(여자면 빨간색, 남자면 파란색) 2. 얼굴검출

    #INPUT_DIR = 'filtered_results_NUDENET_TEST'  # 수집한 데이터 폴더 여자
    INPUT_DIR = 'MALE_DATASETS_300'  # 수집한 데이터 폴더 남자
    OUTPUT_DIR = '_OUT_results_FILTERED'  # 결과 저장 폴더


    # 클래스 인스턴스 생성 및 실행
    experiment = BodybuildingExperiment(
        input_path=INPUT_DIR,
        output_path=OUTPUT_DIR,
        confidence_threshold=0.6,
        detector_type='yolo',  # 'yolo'로 변경 시 기존 파인튜닝 모델 사용
        breast_model_path='best.pt'
    )


    experiment.run_experiment()

    #save_thick_pose_estimation('그림1.png')
