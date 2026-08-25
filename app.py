import os
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import face_recognition
import requests
from PIL import Image
import numpy as np
from urllib.parse import urlparse
import logging
from utils.face_search import AdvancedFaceSearcher
from utils.web_crawler import WebCrawler
from utils.landmark_mapper import LandmarkMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['JSON_SORT_KEYS'] = False

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/landmarks', exist_ok=True)
os.makedirs('exports', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize search engine
searcher = AdvancedFaceSearcher(index_file='data/face_index.pkl')
web_crawler = WebCrawler()
landmark_mapper = LandmarkMapper()

# Load existing index if available
try:
    if os.path.exists('data/face_index.pkl'):
        searcher.load_index()
        logger.info(f"✅ Loaded face index with {len(searcher.known_face_encodings)} faces")
except Exception as e:
    logger.warning(f"Could not load existing index: {e}")

# ============ ROUTES ============

@app.route('/')
def home():
    """Home page - search interface"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Admin dashboard showing index statistics"""
    stats = {
        'total_faces': len(searcher.known_face_encodings),
        'with_landmarks': sum(1 for m in searcher.known_face_metadata if m.get('has_landmarks', False)),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'index_file_exists': os.path.exists('data/face_index.pkl'),
        'index_size_mb': round(os.path.getsize('data/face_index.pkl') / (1024 * 1024), 2) if os.path.exists('data/face_index.pkl') else 0,
        'memory_usage_mb': round(len(searcher.known_face_encodings) * 0.004, 2)
    }
    return render_template('dashboard.html', stats=stats)

@app.route('/search', methods=['POST'])
def search():
    """Search for faces in uploaded image, image URL, or across a specific website"""
    try:
        is_multipart = request.content_type and 'multipart/form-data' in request.content_type

        if is_multipart:
            data = {}
            query_type = request.form.get('type', 'upload')
            use_landmarks = request.form.get('use_landmarks', 'true').lower() == 'true'
            top_k = min(int(request.form.get('top_k', 10)), 50)
        else:
            data = request.get_json() if request.is_json else {}
            query_type = data.get('type', 'upload')
            use_landmarks = data.get('use_landmarks', True)
            top_k = min(int(data.get('top_k', 10)), 50)

        # ============ WEB SEARCH: crawl ONE website, scoped to that domain only ============
        # This never leaves the given site - it only follows same-domain links,
        # and it compares each image it finds directly against the uploaded
        # query image below. It does not search the open internet.
        if query_type == 'web':
            if 'image' not in request.files:
                return jsonify({'error': 'Please upload a reference image to search for', 'code': 'NO_IMAGE'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected', 'code': 'NO_FILE'}), 400

            allowed_ext = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_ext):
                return jsonify({'error': 'Invalid file type. Use JPG, PNG, WEBP, or GIF', 'code': 'INVALID_TYPE'}), 400

            raw_urls = request.form.get('target_urls', '[]')
            try:
                target_urls = json.loads(raw_urls) if isinstance(raw_urls, str) else raw_urls
            except (ValueError, TypeError):
                target_urls = []

            if not target_urls or not isinstance(target_urls, list):
                return jsonify({'error': 'No target website URL provided', 'code': 'NO_URLS'}), 400

            max_pages = min(int(request.form.get('max_pages', 3)), 10)

            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            file.save(temp_path)

            try:
                raw_results = web_crawler.search_websites(
                    target_urls, searcher, query_image=temp_path,
                    max_pages=max_pages, use_landmarks=use_landmarks
                )
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

            formatted_results = [_format_match(m, top_k) for m in raw_results[:top_k]]

            return jsonify({
                'success': True,
                'type': 'web_search',
                'results': formatted_results,
                'count': len(formatted_results),
                'use_landmarks': use_landmarks
            })

        # ============ UPLOAD / URL SEARCH: compare against the saved index ============
        face_data = None
        source = None

        if query_type == 'upload':
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded', 'code': 'NO_IMAGE'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected', 'code': 'NO_FILE'}), 400

            allowed_ext = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_ext):
                return jsonify({'error': 'Invalid file type. Use JPG, PNG, WEBP, or GIF', 'code': 'INVALID_TYPE'}), 400

            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            file.save(temp_path)
            face_data = temp_path
            source = 'upload'

        elif query_type == 'url':
            image_url = (data.get('image_url', '') if not is_multipart else request.form.get('image_url', '')).strip()
            if not image_url:
                return jsonify({'error': 'No image URL provided', 'code': 'NO_URL'}), 400

            if not image_url.startswith(('http://', 'https://')):
                return jsonify({'error': 'Invalid URL format', 'code': 'INVALID_URL'}), 400

            face_data = image_url
            source = 'url'

        else:
            return jsonify({'error': 'Invalid search type', 'code': 'INVALID_TYPE'}), 400

        if len(searcher.known_face_encodings) == 0:
            return jsonify({'error': 'No faces indexed yet. Please index faces first.', 'code': 'EMPTY_INDEX'}), 400

        matches, error = searcher.search_enhanced(
            face_data,
            top_k=top_k,
            use_landmarks=use_landmarks
        )

        if source == 'upload' and os.path.exists(face_data):
            try:
                os.remove(face_data)
            except:
                pass

        if error:
            return jsonify({'error': error, 'code': 'SEARCH_ERROR'}), 400

        formatted_results = [_format_match(m, top_k) for m in matches]

        return jsonify({
            'success': True,
            'type': 'face_search',
            'results': formatted_results,
            'count': len(formatted_results),
            'use_landmarks': use_landmarks
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'code': 'SERVER_ERROR'}), 500


def _format_match(match, top_k=None):
    """Shared result formatting for index-based and web-crawl matches alike."""
    metadata = match.get('metadata', {})
    image_path = metadata.get('image_path', '') or match.get('image_url', '')
    return {
        'similarity': round(float(match['combined_score'] * 100), 2),
        'encoding_similarity': round(float(match['encoding_similarity'] * 100), 2),
        'landmark_similarity': round(float(match['landmark_similarity'] * 100), 2),
        'metadata': metadata,
        'image_path': image_path,
        'image_url': image_path if image_path.startswith(('http://', 'https://')) else '',
        'has_landmarks': match.get('has_landmarks', False),
        'source_page': match.get('source_page', ''),
        'face_index': metadata.get('face_index', 0),
        'indexed_at': metadata.get('indexed_at', '')
    }

@app.route('/index', methods=['POST'])
def index_faces():
    """Index new faces from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'code': 'NO_IMAGE'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'code': 'NO_FILE'}), 400
        
        # Validate file type
        allowed_ext = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_ext):
            return jsonify({'error': 'Invalid file type', 'code': 'INVALID_TYPE'}), 400
        
        # Save temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        # Get metadata
        name = request.form.get('name', f'Face_{len(searcher.known_face_encodings) + 1}').strip()
        tags = request.form.get('tags', '').strip()
        
        # Index the face
        success, msg = searcher.index_face_enhanced(temp_path, {
            'name': name,
            'tags': [t.strip() for t in tags.split(',') if t.strip()],
            'indexed_at': datetime.now().isoformat()
        })
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        if success:
            searcher.save_index()
            return jsonify({
                'success': True,
                'message': msg,
                'total_faces': len(searcher.known_face_encodings)
            })
        else:
            return jsonify({'error': msg, 'code': 'INDEX_ERROR'}), 400
            
    except Exception as e:
        logger.error(f"Index error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'code': 'SERVER_ERROR'}), 500

@app.route('/compare', methods=['POST'])
def compare_faces():
    """Compare two faces directly"""
    try:
        data = request.get_json()
        if not data or 'face1' not in data or 'face2' not in data:
            return jsonify({'error': 'Two face URLs/paths required', 'code': 'MISSING_FACES'}), 400
        
        face1 = data['face1'].strip()
        face2 = data['face2'].strip()
        
        if not face1 or not face2:
            return jsonify({'error': 'Face paths cannot be empty', 'code': 'EMPTY_FACES'}), 400
        
        comparison = searcher.compare_two_faces(face1, face2)
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        logger.error(f"Compare error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'code': 'SERVER_ERROR'}), 500

@app.route('/visualize', methods=['POST'])
def visualize_landmarks():
    """Generate landmark visualization for a face"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Image URL/path required', 'code': 'NO_IMAGE'}), 400
        
        image_path = data['image'].strip()
        output_path = f"static/landmarks/{uuid.uuid4()}_landmarks.jpg"
        
        success, result = landmark_mapper.visualize_landmarks(image_path, output_path)
        
        if success:
            return jsonify({
                'success': True,
                'visualization_url': url_for('static', filename=result.replace('static/', ''))
            })
        else:
            return jsonify({'error': result, 'code': 'VIZ_ERROR'}), 400
            
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'code': 'SERVER_ERROR'}), 500

@app.route('/stats')
def get_stats():
    """Get search statistics"""
    stats = {
        'total_faces': len(searcher.known_face_encodings),
        'with_landmarks': sum(1 for m in searcher.known_face_metadata if m.get('has_landmarks', False)),
        'index_size_mb': round(os.path.getsize('data/face_index.pkl') / (1024 * 1024), 2) if os.path.exists('data/face_index.pkl') else 0
    }
    return jsonify(stats)

@app.route('/export', methods=['POST'])
def export_results():
    """Export search results as JSON"""
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({'error': 'No results provided', 'code': 'NO_RESULTS'}), 400
        
        results = data['results']
        export_file = f"exports/search_{uuid.uuid4()}.json"
        
        with open(export_file, 'w') as f:
            json.dump({
                'exported_at': datetime.now().isoformat(),
                'results_count': len(results),
                'results': results
            }, f, indent=2)
        
        return send_file(export_file, as_attachment=True, download_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'code': 'SERVER_ERROR'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'faces_indexed': len(searcher.known_face_encodings),
        'version': '1.0.0'
    })

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'code': 'NOT_FOUND'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB', 'code': 'FILE_TOO_LARGE'}), 413

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error', 'code': 'SERVER_ERROR'}), 500

# ============ MAIN ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
