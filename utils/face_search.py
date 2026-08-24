import os
import pickle
import numpy as np
from PIL import Image
import face_recognition
import requests
from io import BytesIO
from scipy.spatial import distance
import cv2
import logging

logger = logging.getLogger(__name__)

class AdvancedFaceSearcher:
    def __init__(self, index_file='data/face_index.pkl'):
        self.index_file = index_file
        self.known_face_encodings = []
        self.known_face_landmarks = []
        self.known_face_metadata = []
        self.tolerance = 0.6
        self.landmark_weight = 0.3
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.timeout = 10
    
    def load_index(self):
        """Load face index from disk"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_landmarks = data.get('landmarks', [])
                    self.known_face_metadata = data.get('metadata', [])
                logger.info(f"✅ Loaded {len(self.known_face_encodings)} faces from index")
                return True
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                return False
        return False
    
    def save_index(self):
        """Save face index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_file) or '.', exist_ok=True)
            with open(self.index_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'landmarks': self.known_face_landmarks,
                    'metadata': self.known_face_metadata
                }, f)
            logger.info(f"✅ Saved index with {len(self.known_face_encodings)} faces")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def get_enhanced_face_data(self, image_path):
        """Extract both encoding and landmarks"""
        try:
            # Load image
            if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
                response = self.session.get(image_path, timeout=10)
                if response.status_code != 200:
                    return None, f"Failed to download image (HTTP {response.status_code})"
                image = face_recognition.load_image_file(BytesIO(response.content))
            else:
                if not os.path.exists(image_path):
                    return None, "Image file not found"
                image = face_recognition.load_image_file(image_path)
            
            # Get face locations
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return None, "No face found in image"
            
            # Get encoding
            encodings = face_recognition.face_encodings(image, face_locations)
            
            # Get landmarks
            landmarks, error = self.extract_face_landmarks(image)
            if error:
                landmarks = None
            
            return {
                'encodings': encodings,
                'landmarks': landmarks,
                'face_locations': face_locations
            }, None
            
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
    
    def extract_face_landmarks(self, image):
        """Extract 68-point facial landmarks"""
        try:
            face_landmarks_list = face_recognition.face_landmarks(image)
            if not face_landmarks_list:
                return None, "No face landmarks found"
            
            landmarks = {}
            for feature, points in face_landmarks_list[0].items():
                landmarks[feature] = np.array(points)
            
            # Calculate metrics
            metrics = self.calculate_face_metrics(landmarks)
            landmarks['metrics'] = metrics
            
            return landmarks, None
            
        except Exception as e:
            return None, f"Error extracting landmarks: {str(e)}"
    
    def calculate_face_metrics(self, landmarks):
        """Calculate facial proportions"""
        metrics = {}
        try:
            left_eye = landmarks.get('left_eye', [])
            right_eye = landmarks.get('right_eye', [])
            
            if len(left_eye) > 0 and len(right_eye) > 0:
                left_width = distance.euclidean(left_eye[0], left_eye[3])
                left_height = distance.euclidean(left_eye[1], left_eye[5])
                right_width = distance.euclidean(right_eye[0], right_eye[3])
                right_height = distance.euclidean(right_eye[1], right_eye[5])
                
                metrics['left_eye_ratio'] = float(left_height / left_width) if left_width > 0 else 0.0
                metrics['right_eye_ratio'] = float(right_height / right_width) if right_width > 0 else 0.0
            
            # Inter-eye distance
            if len(left_eye) > 0 and len(right_eye) > 0:
                left_center = np.mean(left_eye, axis=0)
                right_center = np.mean(right_eye, axis=0)
                metrics['inter_eye_distance'] = float(distance.euclidean(left_center, right_center))
            
        except Exception as e:
            logger.warning(f"Error calculating face metrics: {e}")
        
        return metrics
    
    def compute_landmark_similarity(self, landmarks1, landmarks2):
        """Compare two face landmarks"""
        if not landmarks1 or not landmarks2:
            return 0.0
        
        scores = []
        weights = {
            'left_eye': 1.2,
            'right_eye': 1.2,
            'nose_tip': 1.0,
            'top_lip': 0.8,
            'bottom_lip': 0.8,
            'chin': 0.6
        }
        
        for feature, weight in weights.items():
            points1 = landmarks1.get(feature)
            points2 = landmarks2.get(feature)
            
            if points1 is None or points2 is None:
                continue
            
            min_len = min(len(points1), len(points2))
            if min_len == 0:
                continue
            
            try:
                centroid1 = np.mean(points1[:min_len], axis=0)
                centroid2 = np.mean(points2[:min_len], axis=0)
                
                aligned1 = points1[:min_len] - centroid1
                aligned2 = points2[:min_len] - centroid2
                
                scale1 = np.max(np.linalg.norm(aligned1, axis=1))
                scale2 = np.max(np.linalg.norm(aligned2, axis=1))
                
                if scale1 > 0:
                    aligned1 = aligned1 / scale1
                if scale2 > 0:
                    aligned2 = aligned2 / scale2
                
                diff = np.linalg.norm(aligned1 - aligned2)
                max_dist = np.sqrt(2) * 2
                
                similarity = max(0.0, 1.0 - (diff / max_dist))
                scores.append(float(similarity * weight))
                
            except Exception as e:
                logger.warning(f"Error comparing landmark {feature}: {e}")
                continue
        
        if not scores:
            return 0.0
        
        total_weight = sum(weights.values())
        return float(sum(scores) / total_weight)
    
    def index_face_enhanced(self, image_path, metadata=None):
        """Index a face with both encoding and landmarks"""
        data, error = self.get_enhanced_face_data(image_path)
        if error:
            return False, error
        
        for i, encoding in enumerate(data['encodings']):
            self.known_face_encodings.append(encoding)
            
            if data['landmarks']:
                self.known_face_landmarks.append(data['landmarks'])
            else:
                self.known_face_landmarks.append(None)
            
            meta = metadata or {}
            meta.update({
                'image_path': image_path,
                'face_index': i,
                'has_landmarks': data['landmarks'] is not None
            })
            self.known_face_metadata.append(meta)
        
        return True, f"Indexed {len(data['encodings'])} face(s)"
    
    def search_enhanced(self, query_image, top_k=10, use_landmarks=True):
        """Search using both encoding and landmarks"""
        if not self.known_face_encodings:
            return [], "No faces in index"
        
        query_data, error = self.get_enhanced_face_data(query_image)
        if error:
            return [], error
        
        if not query_data['encodings']:
            return [], "No face found in query image"
        
        query_encoding = query_data['encodings'][0]
        query_landmarks = query_data['landmarks']
        
        combined_scores = []
        
        for idx, (known_encoding, known_landmarks) in enumerate(
            zip(self.known_face_encodings, self.known_face_landmarks)
        ):
            # Face encoding similarity
            encoding_distance = face_recognition.face_distance([known_encoding], query_encoding)[0]
            encoding_similarity = float(1.0 - min(float(encoding_distance), 1.0))
            
            # Landmark similarity
            landmark_similarity = 0.0
            if use_landmarks and query_landmarks and known_landmarks:
                landmark_similarity = self.compute_landmark_similarity(
                    query_landmarks, known_landmarks
                )
            
            # Combined score
            if use_landmarks and query_landmarks and known_landmarks:
                combined_score = float(
                    (1.0 - self.landmark_weight) * encoding_similarity +
                    self.landmark_weight * landmark_similarity
                )
            else:
                combined_score = encoding_similarity
            
            if combined_score >= (1.0 - self.tolerance):
                combined_scores.append({
                    'index': idx,
                    'encoding_similarity': encoding_similarity,
                    'landmark_similarity': landmark_similarity,
                    'combined_score': combined_score,
                    'metadata': self.known_face_metadata[idx],
                    'has_landmarks': known_landmarks is not None
                })
        
        combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Remove duplicates
        seen_paths = set()
        unique_results = []
        for match in combined_scores:
            path = match['metadata'].get('image_path', '')
            if path not in seen_paths:
                seen_paths.add(path)
                unique_results.append(match)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results, None
    
    def compare_two_faces(self, face1_path, face2_path):
        """Compare two faces directly"""
        data1, error1 = self.get_enhanced_face_data(face1_path)
        data2, error2 = self.get_enhanced_face_data(face2_path)
        
        if error1 or error2:
            return {"error": error1 or error2}
        
        if not data1['encodings'] or not data2['encodings']:
            return {"error": "No face found in one or both images"}
        
        encoding1 = data1['encodings'][0]
        encoding2 = data2['encodings'][0]
        
        encoding_dist = face_recognition.face_distance([encoding1], encoding2)[0]
        encoding_sim = float(1.0 - min(float(encoding_dist), 1.0))
        
        landmark_sim = 0.0
        if data1['landmarks'] and data2['landmarks']:
            landmark_sim = self.compute_landmark_similarity(
                data1['landmarks'], data2['landmarks']
            )
        
        combined = float(
            (1.0 - self.landmark_weight) * encoding_sim +
            self.landmark_weight * landmark_sim
        )
        
        return {
            "encoding_similarity": round(encoding_sim, 4),
            "landmark_similarity": round(landmark_sim, 4),
            "combined_score": round(combined, 4),
            "is_match": combined >= (1.0 - self.tolerance),
            "landmarks1_available": data1['landmarks'] is not None,
            "landmarks2_available": data2['landmarks'] is not None
        }
