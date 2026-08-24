import os
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class LandmarkMapper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def load_image(self, image_path):
        """Load image from path or URL"""
        try:
            if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
                response = self.session.get(image_path, timeout=10)
                if response.status_code != 200:
                    return None, "Failed to download image"
                return Image.open(BytesIO(response.content)), None
            else:
                if not os.path.exists(image_path):
                    return None, "Image file not found"
                return Image.open(image_path), None
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
    
    def visualize_landmarks(self, image_path, output_path):
        """Draw facial landmarks on image"""
        try:
            image_pil, error = self.load_image(image_path)
            if error:
                return False, error
            
            # Convert to numpy array for face_recognition
            image_np = np.array(image_pil)
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np] * 3, axis=-1)
            
            # Get landmarks
            face_landmarks_list = face_recognition.face_landmarks(image_np)
            if not face_landmarks_list:
                return False, "No face landmarks found"
            
            # Draw on image
            draw = ImageDraw.Draw(image_pil)
            
            # Colors for different features
            colors = {
                'chin': 'red',
                'left_eyebrow': 'blue',
                'right_eyebrow': 'blue',
                'nose_bridge': 'green',
                'nose_tip': 'green',
                'left_eye': 'cyan',
                'right_eye': 'cyan',
                'top_lip': 'magenta',
                'bottom_lip': 'magenta'
            }
            
            for face_landmarks in face_landmarks_list:
                for feature, points in face_landmarks.items():
                    color = colors.get(feature, 'yellow')
                    points_list = [(point[0], point[1]) for point in points]
                    
                    # Draw circles at each point
                    for point in points_list:
                        draw.ellipse(
                            [point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2],
                            fill=color,
                            outline=color
                        )
                    
                    # Draw lines connecting points
                    if len(points_list) > 1:
                        draw.line(points_list + [points_list[0]], fill=color, width=2)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Save
            image_pil.save(output_path, quality=90)
            logger.info(f"Saved landmark visualization to {output_path}")
            return True, output_path
            
        except Exception as e:
            logger.error(f"Error visualizing landmarks: {e}")
            return False, f"Error: {str(e)}"
