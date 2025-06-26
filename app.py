from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import uuid
import threading
from queue import Queue
import time
from datetime import datetime
import zipfile
import tempfile
import gc

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# สร้างโฟลเดอร์หากยังไม่มี
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load YOLOv8 model (ใส่ path ของ model ที่ train แล้ว)
MODEL_PATH = 'best.pt'  # เปลี่ยนเป็น path ของ model ของคุณ

# Model Manager สำหรับจัดการหน่วยความจำ
class ModelManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.last_used = time.time()
        self.lock = threading.Lock()
    
    def get_model(self):
        with self.lock:
            if self.model is None:
                self.model = YOLO(self.model_path)
            self.last_used = time.time()
            return self.model
    
    def cleanup_if_unused(self, timeout=300):  # 5 minutes
        with self.lock:
            if time.time() - self.last_used > timeout:
                self.model = None
                gc.collect()

# สร้าง model manager
model_manager = ModelManager(MODEL_PATH)

# กำหนด file extensions ที่รองรับ
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, confidence_threshold=0.5):
    """
    ประมวลผลรูปภาพด้วย YOLOv8
    """
    try:
        # ใช้ model จาก manager
        model = model_manager.get_model()
        
        # อ่านรูปภาพ
        image = cv2.imread(image_path)
        if image is None:
            return None, "ไม่สามารถอ่านรูปภาพได้"
        
        # ปรับขนาดรูปภาพถ้าใหญ่เกินไป
        height, width = image.shape[:2]
        max_size = 1280
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # ทำ inference
        results = model(image, conf=confidence_threshold, verbose=False)
        
        # วาดผลลัพธ์บนรูปภาพ
        annotated_image = results[0].plot()
        
        # บันทึกรูปภาพผลลัพธ์
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # สร้างข้อมูลสำหรับแสดงผล
        detections = []
        for box in results[0].boxes:
            if box.conf[0] >= confidence_threshold:
                detection = {
                    'class': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return {
            'result_image': result_filename,
            'detections': detections,
            'total_detections': len(detections)
        }, None
        
    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {str(e)}"

# Task Queue สำหรับประมวลผลแบบ Async
class TaskQueue:
    def __init__(self):
        self.queue = Queue()
        self.results = {}
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        while True:
            task_data = self.queue.get()
            if task_data is None:
                break
            
            task_id, image_path, confidence = task_data
            try:
                result, error = process_image(image_path, confidence)
                self.results[task_id] = {'result': result, 'error': error, 'completed': True}
                # ลบไฟล์ต้นฉบับ
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                self.results[task_id] = {'result': None, 'error': str(e), 'completed': True}
                if os.path.exists(image_path):
                    os.remove(image_path)
            finally:
                self.queue.task_done()
    
    def add_task(self, task_id, image_path, confidence):
        self.results[task_id] = {'completed': False}
        self.queue.put((task_id, image_path, confidence))
    
    def get_result(self, task_id):
        return self.results.get(task_id)
    
    def remove_result(self, task_id):
        self.results.pop(task_id, None)

# สร้าง task queue
task_queue = TaskQueue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์'}), 400
    
    file = request.files['file']
    confidence = float(request.form.get('confidence', 0.5))
    
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
    
    if file and allowed_file(file.filename):
        # บันทึกไฟล์
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # ประมวลผลรูปภาพ
        result, error = process_image(file_path, confidence)
        
        # ลบไฟล์ต้นฉบับ
        os.remove(file_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    return jsonify({'error': 'ไฟล์ไม่ถูกต้อง'}), 400

@app.route('/result/<filename>')
def get_result_image(filename):
    """ส่งรูปภาพผลลัพธ์"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/webcam', methods=['POST'])
def process_webcam():
    """ประมวลผลรูปภาพจาก webcam"""
    try:
        # รับข้อมูลรูปภาพจาก base64
        image_data = request.json['image']
        confidence = float(request.json.get('confidence', 0.5))
        
        # แปลง base64 เป็นรูปภาพ
        image_data = image_data.split(',')[1]  # ตัด data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # บันทึกเป็นไฟล์ชั่วคราว
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        image.save(temp_path)
        
        # ประมวลผล
        result, error = process_image(temp_path, confidence)
        
        # ลบไฟล์ชั่วคราว
        os.remove(temp_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """ประมวลผลรูปภาพหลายรูปพร้อมกัน"""
    if 'files' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์'}), 400
    
    files = request.files.getlist('files')
    confidence = float(request.form.get('confidence', 0.5))
    
    if not files or len(files) == 0:
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
    
    results = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # บันทึกไฟล์
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # ประมวลผลรูปภาพ
                result, error = process_image(file_path, confidence)
                
                # ลบไฟล์ต้นฉบับ
                os.remove(file_path)
                
                if error:
                    errors.append(f"{filename}: {error}")
                else:
                    result['filename'] = filename
                    results.append(result)
                    
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
        else:
            errors.append(f"{file.filename}: ไฟล์ไม่ถูกต้อง")
    
    return jsonify({
        'success': len(results) > 0,
        'results': results,
        'errors': errors,
        'processed_count': len(results),
        'error_count': len(errors)
    })

@app.route('/async_upload', methods=['POST'])
def async_upload():
    """อัปโหลดและประมวลผลแบบ Asynchronous"""
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์'}), 400
    
    file = request.files['file']
    confidence = float(request.form.get('confidence', 0.5))
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'ไฟล์ไม่ถูกต้อง'}), 400
    
    # บันทึกไฟล์
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    # สร้าง task ID และเพิ่มเข้า queue
    task_id = str(uuid.uuid4())
    task_queue.add_task(task_id, file_path, confidence)
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'เริ่มประมวลผลแล้ว ใช้ task_id เพื่อตรวจสอบผลลัพธ์'
    })

@app.route('/check_task/<task_id>')
def check_task(task_id):
    """ตรวจสอบสถานะการประมวลผล"""
    result = task_queue.get_result(task_id)
    
    if not result:
        return jsonify({'error': 'ไม่พบ task'}), 404
    
    if not result['completed']:
        return jsonify({'status': 'processing', 'completed': False})
    
    response = {
        'status': 'completed',
        'completed': True,
        'success': result['error'] is None,
        'result': result['result'],
        'error': result['error']
    }
    
    # ลบผลลัพธ์จาก memory หลังส่งกลับ
    task_queue.remove_result(task_id)
    
    return jsonify(response)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint สำหรับการตรวจจับวัตถุ (JSON response)"""
    try:
        if 'image' in request.files:
            # ประมวลผลจากไฟล์
            file = request.files['image']
            confidence = float(request.form.get('confidence', 0.5))
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'ไฟล์ไม่ถูกต้อง'}), 400
            
            # บันทึกและประมวลผล
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            result, error = process_image(file_path, confidence)
            os.remove(file_path)  # ลบไฟล์ชั่วคราว
            
        elif 'image_base64' in request.json:
            # ประมวลผลจาก base64
            image_data = request.json['image_base64']
            confidence = float(request.json.get('confidence', 0.5))
            
            # แปลง base64 เป็นรูปภาพ
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # บันทึกเป็นไฟล์ชั่วคราว
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            image.save(temp_path)
            
            result, error = process_image(temp_path, confidence)
            os.remove(temp_path)  # ลบไฟล์ชั่วคราว
            
        else:
            return jsonify({'error': 'ไม่พบข้อมูลรูปภาพ'}), 400
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'detections': result['detections'],
            'total_detections': result['total_detections'],
            'result_image_url': f"/result/{result['result_image']}"
        })
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/model/classes')
def api_model_classes():
    """API สำหรับดึงรายชื่อ classes ที่ model รองรับ"""
    try:
        model = model_manager.get_model()
        return jsonify({
            'classes': model.names,
            'num_classes': len(model.names)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info')
def model_info():
    """ข้อมูลเกี่ยวกับ model"""
    try:
        model = model_manager.get_model()
        return jsonify({
            'model_name': MODEL_PATH,
            'classes': model.names,
            'num_classes': len(model.names)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_results')
def download_results():
    """ดาวน์โหลดผลลัพธ์ทั้งหมดเป็น ZIP"""
    try:
        # สร้างไฟล์ ZIP ชั่วคราว
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'results.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename in os.listdir(app.config['RESULTS_FOLDER']):
                file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
                if os.path.isfile(file_path):
                    zipf.write(file_path, filename)
        
        return send_file(zip_path, as_attachment=True, download_name='yolo_results.zip')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cleanup functions
def cleanup_old_files():
    """ลบไฟล์ผลลัพธ์ที่เก่าเกิน 1 ชั่วโมง"""
    current_time = time.time()
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        try:
                            os.remove(file_path)
                        except:
                            pass

def start_cleanup_scheduler():
    def cleanup_worker():
        while True:
            time.sleep(1800)  # 30 minutes
            cleanup_old_files()
            model_manager.cleanup_if_unused()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

# เริ่ม cleanup scheduler
start_cleanup_scheduler()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)