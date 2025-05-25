# app.py - Flask 백엔드 (수정된 버전)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import json
import threading
import uuid
from datetime import datetime
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
)
from datasets import Dataset as HFDataset
import logging
from typing import Dict, List, Optional
import zipfile
import io
import traceback

from werkzeug.utils import secure_filename

# Flask 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
CORS(app)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# 업로드 폴더 설정
UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 전역 변수
training_jobs = {}
model_cache = {}

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebFineTuner:
    """웹 기반 파인튜너"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 사용 가능한 한국어 모델들
        self.available_models = {
            "polyglot-ko-1.3b": "EleutherAI/polyglot-ko-1.3b",
            "kogpt": "kakaobrain/kogpt",
            "kobert": "monologg/kobert",
            "kullm-polyglot": "nlpai-lab/kullm-polyglot-5.8b-v2",
            "llama-ko-7b": "beomi/llama-2-ko-7b"
        }

    def emit_progress(self, message: str, progress: int = 0):
        """진행상황 전송"""
        socketio.emit('training_progress', {
            'job_id': self.job_id,
            'message': message,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        })

    def load_model(self, model_name: str, task_type: str):
        """모델 로드"""
        self.emit_progress(f"모델 로딩 중: {model_name}", 10)

        try:
            model_path = self.available_models.get(model_name, model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            if task_type == "generation":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    trust_remote_code=True
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.emit_progress("모델 로딩 완료", 20)

        except Exception as e:
            self.emit_progress(f"모델 로딩 실패: {str(e)}", 0)
            raise

    def prepare_dataset(self, data: List[Dict], task_type: str, max_length: int):
        """데이터셋 준비"""
        self.emit_progress("데이터셋 준비 중...", 30)

        if task_type == "generation":
            def tokenize_function(examples):
                texts = [
                    f"질문: {inp}\n답변: {out}{self.tokenizer.eos_token}"
                    for inp, out in zip(examples["input"], examples["output"])
                ]
                model_inputs = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                model_inputs["labels"] = model_inputs["input_ids"].copy()
                return model_inputs

            dataset_dict = {
                "input": [item["input"] for item in data],
                "output": [item["output"] for item in data]
            }

        else:  # classification, regression 등
            def tokenize_function(examples):
                model_inputs = self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                model_inputs["labels"] = examples["labels"]
                return model_inputs

            dataset_dict = {
                "text": [item["text"] for item in data],
                "labels": [item["label"] for item in data]
            }

        dataset = HFDataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        self.emit_progress("데이터셋 준비 완료", 40)
        return tokenized_dataset

    def train_model(self, config: Dict):
        """모델 훈련"""
        try:
            # 모델 로드
            self.load_model(config['model_name'], config['task_type'])

            # 데이터 준비
            train_dataset = self.prepare_dataset(
                config['train_data'],
                config['task_type'],
                config['max_length']
            )

            val_dataset = None
            if config.get('val_data'):
                val_dataset = self.prepare_dataset(
                    config['val_data'],
                    config['task_type'],
                    config['max_length']
                )

            # 데이터 콜레이터 설정
            if config['task_type'] == "generation":
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False,
                )
            else:
                data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # 훈련 설정
            output_dir = f"./models/{self.job_id}"
            os.makedirs(output_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config['num_epochs'],
                per_device_train_batch_size=config['batch_size'],
                per_device_eval_batch_size=config['batch_size'],
                warmup_steps=config.get('warmup_steps', 100),
                weight_decay=config.get('weight_decay', 0.01),
                learning_rate=config['learning_rate'],
                logging_dir=f"{output_dir}/logs",
                logging_steps=config.get('logging_steps', 50),
                evaluation_strategy="steps" if val_dataset else "no",
                eval_steps=config.get('eval_steps', 100) if val_dataset else None,
                save_steps=config.get('save_steps', 200),
                save_total_limit=2,
                load_best_model_at_end=True if val_dataset else False,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=config.get('gradient_checkpointing', True),
                dataloader_pin_memory=False,
                report_to=[],
            )

            self.emit_progress("훈련 시작...", 50)

            # 훈련 진행상황 추적을 위한 커스텀 콜백
            class ProgressCallback(TrainerCallback):
                def __init__(self, emitter, job_id):
                    self.emitter = emitter
                    self.job_id = job_id
                    self.total_steps = None

                def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                    self.total_steps = state.max_steps or (
                                args.num_train_epochs * state.num_examples // args.per_device_train_batch_size)
                    self.emitter.emit_progress("훈련 시작", 50)

                def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                    if self.total_steps:
                        step_progress = (state.global_step / self.total_steps)
                        progress_percent = int(50 + step_progress * 50)
                        self.emitter.emit_progress(f"{state.global_step}/{self.total_steps} 스텝 진행 중", progress_percent)

                def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                    self.emitter.emit_progress(f"에포크 {int(state.epoch)}/{int(args.num_train_epochs)} 완료", None)

                def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                    self.emitter.emit_progress("훈련 종료", 100)

            # 트레이너 설정
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[
                    ProgressCallback(self, self.job_id),
                    EarlyStoppingCallback(early_stopping_patience=3)
                ] if val_dataset else [ProgressCallback(self, self.job_id)],
            )



            trainer.add_callback(ProgressCallback(self, self.job_id))

            # 훈련 실행
            trainer.train()

            # 모델 저장
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            # 모델 정보 저장
            model_info = {
                'job_id': self.job_id,
                'model_name': config['model_name'],
                'task_type': config['task_type'],
                'config': config,
                'created_at': datetime.now().isoformat(),
                'model_path': output_dir
            }

            with open(f"{output_dir}/model_info.json", 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)

            self.emit_progress("훈련 완료!", 100)

            return {
                'success': True,
                'model_path': output_dir,
                'model_info': model_info
            }

        except Exception as e:
            self.emit_progress(f"훈련 실패: {str(e)}", 0)
            return {'success': False, 'error': str(e)}


# 파일 형식 검증 함수
def allowed_file(filename):
    """허용된 파일 형식인지 확인"""
    ALLOWED_EXTENSIONS = {'json', 'csv', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_uploaded_file(file_path):
    """업로드된 파일을 파싱"""
    try:
        file_extension = file_path.rsplit('.', 1)[1].lower()

        if file_extension == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # JSON 형식 검증 및 변환
            if isinstance(data, list):
                # 이미 리스트 형태인 경우
                parsed_data = data
            elif isinstance(data, dict):
                # 딕셔너리인 경우 리스트로 변환
                if 'data' in data:
                    parsed_data = data['data']
                else:
                    parsed_data = [data]
            else:
                raise ValueError("지원하지 않는 JSON 형식입니다.")

        elif file_extension == 'csv':
            # CSV 파일 처리
            df = pd.read_csv(file_path, encoding='utf-8')
            parsed_data = df.to_dict('records')

        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_extension}")

        # 데이터 형식 검증
        if not parsed_data:
            raise ValueError("빈 데이터 파일입니다.")

        # 각 항목이 input, output 키를 가지고 있는지 확인
        for i, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                raise ValueError(f"행 {i + 1}: 딕셔너리 형태가 아닙니다.")
            if 'input' not in item or 'output' not in item:
                raise ValueError(f"행 {i + 1}: 'input' 또는 'output' 키가 없습니다.")

        return parsed_data

    except Exception as e:
        logger.error(f"파일 파싱 오류: {str(e)}")
        raise


# API 엔드포인트들
@app.route('/api/models', methods=['GET'])
def get_available_models():
    """사용 가능한 모델 목록"""
    models = {
        "polyglot-ko-1.3b": {"name": "Polyglot Korean 1.3B", "type": "generation"},
        "kogpt": {"name": "KaKao KoGPT", "type": "generation"},
        "kobert": {"name": "KoBERT", "type": "classification"},
        "kullm-polyglot": {"name": "KULLM Polyglot", "type": "generation"},
        "llama-ko-7b": {"name": "Llama Korean 7B", "type": "generation"}
    }
    return jsonify(models)


@app.route("/api/upload-data", methods=["POST"])
def upload_data():
    """파일 업로드 및 데이터 처리"""
    try:
        # 파일 존재 확인
        if "file" not in request.files:
            return jsonify(success=False, error="파일이 포함되지 않았습니다."), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify(success=False, error="파일명이 비어 있습니다."), 400

        # 파일 형식 확인
        if not allowed_file(file.filename):
            return jsonify(success=False, error="지원하지 않는 파일 형식입니다. (JSON, CSV만 지원)"), 400

        # 안전한 파일명 생성
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_{uuid.uuid4().hex[:8]}.json"

        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # 파일 저장
        file.save(file_path)
        logger.info(f"파일 저장됨: {file_path}")

        # 파일 파싱
        try:
            parsed_data = parse_uploaded_file(file_path)
            logger.info(f"데이터 파싱 완료: {len(parsed_data)}개 항목")

            return jsonify({
                'success': True,
                'data': parsed_data,
                'total_rows': len(parsed_data),
                'columns': ['input', 'output'],
                'filename': filename
            })

        except Exception as parse_error:
            logger.error(f"파일 파싱 오류: {str(parse_error)}")
            return jsonify(success=False, error=f"파일 파싱 오류: {str(parse_error)}"), 400

    except Exception as e:
        logger.error(f"파일 업로드 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(success=False, error=f"서버 오류: {str(e)}"), 500


@app.route('/api/start-training', methods=['POST'])
def start_training():
    """훈련 시작"""
    try:
        config = request.get_json()
        if not config:
            return jsonify({'error': '요청 데이터가 없습니다'}), 400

        # 필수 필드 확인
        required_fields = ['model_name', 'task_type', 'train_data']
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'필수 필드 누락: {field}'}), 400

        job_id = str(uuid.uuid4())

        # 훈련 작업 정보 저장
        training_jobs[job_id] = {
            'status': 'starting',
            'config': config,
            'created_at': datetime.now().isoformat()
        }

        # 백그라운드에서 훈련 실행
        def run_training():
            try:
                finetuner = WebFineTuner(job_id)
                result = finetuner.train_model(config)
                training_jobs[job_id]['result'] = result
                training_jobs[job_id]['status'] = 'completed' if result['success'] else 'failed'
            except Exception as e:
                logger.error(f"훈련 실행 오류: {str(e)}")
                training_jobs[job_id]['result'] = {'success': False, 'error': str(e)}
                training_jobs[job_id]['status'] = 'failed'

        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id
        })

    except Exception as e:
        logger.error(f"훈련 시작 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training-status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """훈련 상태 조회"""
    job = training_jobs.get(job_id)
    if not job:
        return jsonify({'error': '작업을 찾을 수 없습니다'}), 404

    return jsonify(job)


@app.route('/api/jobs', methods=['GET'])
def get_all_jobs():
    """전체 Job 목록 조회"""
    try:
        # 쿼리 파라미터 처리
        status_filter = request.args.get('status')  # 상태 필터링 (예: completed, running, failed)
        limit = request.args.get('limit', type=int)  # 결과 제한 수
        sort_by = request.args.get('sort_by', 'created_at')  # 정렬 기준 (created_at, status)
        sort_order = request.args.get('sort_order', 'desc')  # 정렬 순서 (asc, desc)

        # 모든 Job 데이터 가져오기
        jobs_list = []
        for job_id, job_data in training_jobs.items():
            job_info = {
                'job_id': job_id,
                'status': job_data.get('status', 'unknown'),
                'created_at': job_data.get('created_at'),
                'config': job_data.get('config', {}),
                'result': job_data.get('result')
            }

            # 추가 정보 추출
            config = job_data.get('config', {})
            job_info.update({
                'model_name': config.get('model_name'),
                'task_type': config.get('task_type'),
                'dataset_name': config.get('dataset_name'),
                # 진행률 계산 (실제 구현에서는 WebFineTuner에서 가져와야 함)
                'progress': get_job_progress(job_id, job_data),
                # 시작/종료 시간
                'start_time': job_data.get('start_time'),
                'end_time': job_data.get('end_time'),
                # 로그 정보 (최근 몇 개만)
                'recent_logs': get_recent_logs(job_id, limit=5)
            })

            jobs_list.append(job_info)

        # 상태 필터링
        if status_filter:
            jobs_list = [job for job in jobs_list if job['status'] == status_filter]

        # 정렬
        reverse_order = sort_order.lower() == 'desc'
        if sort_by == 'created_at':
            jobs_list.sort(key=lambda x: x.get('created_at', ''), reverse=reverse_order)
        elif sort_by == 'status':
            jobs_list.sort(key=lambda x: x.get('status', ''), reverse=reverse_order)

        # 결과 제한
        if limit:
            jobs_list = jobs_list[:limit]

        return jsonify({
            'success': True,
            'jobs': jobs_list,
            'total_count': len(training_jobs),
            'filtered_count': len(jobs_list)
        })

    except Exception as e:
        logger.error(f"Job 목록 조회 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


def get_job_progress(job_id, job_data):
    """Job 진행률 계산"""
    try:
        status = job_data.get('status', 'unknown')

        # 상태별 기본 진행률
        status_progress = {
            'starting': 5.0,
            'preparing': 10.0,
            'training': 50.0,  # 실제로는 WebFineTuner에서 실시간 진행률 가져와야 함
            'validating': 85.0,
            'saving': 95.0,
            'completed': 100.0,
            'failed': 0.0,
            'cancelled': 0.0
        }

        # 실제 구현에서는 WebFineTuner 인스턴스에서 실시간 진행률을 가져와야 함
        # 예: finetuner = get_finetuner_instance(job_id)
        # return finetuner.get_progress() if finetuner else status_progress.get(status, 0.0)

        return status_progress.get(status, 0.0)

    except Exception as e:
        logger.error(f"진행률 계산 오류: {str(e)}")
        return 0.0


def get_recent_logs(job_id, limit=5):
    """최근 로그 가져오기"""
    try:
        # 실제 구현에서는 로그 파일이나 로그 저장소에서 가져와야 함
        # 예시 로그 데이터
        job_data = training_jobs.get(job_id, {})
        status = job_data.get('status', 'unknown')

        # 상태별 샘플 로그 (실제로는 실시간 로그를 저장하고 조회해야 함)
        sample_logs = {
            'starting': [
                {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '훈련 작업 초기화 중...'},
                {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '데이터셋 로딩 시작'}
            ],
            'training': [
                {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '에포크 1/10 시작'},
                {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '배치 100/500 처리 완료'},
                {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '손실값: 0.342'}
            ],
            'completed': [
                {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '모델 저장 완료'},
                {'timestamp': datetime.now().isoformat(), 'level': 'SUCCESS', 'message': '훈련이 성공적으로 완료되었습니다'}
            ],
            'failed': [
                {'timestamp': datetime.now().isoformat(), 'level': 'ERROR', 'message': '훈련 중 오류 발생'},
                {'timestamp': datetime.now().isoformat(), 'level': 'ERROR',
                 'message': job_data.get('result', {}).get('error', '알 수 없는 오류')}
            ]
        }

        logs = sample_logs.get(status, [])
        return logs[-limit:] if logs else []

    except Exception as e:
        logger.error(f"로그 조회 오류: {str(e)}")
        return []


@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Job 삭제"""
    try:
        if job_id not in training_jobs:
            return jsonify({'error': '작업을 찾을 수 없습니다'}), 404

        job_data = training_jobs[job_id]
        status = job_data.get('status')

        # 실행 중인 작업은 삭제 불가
        if status in ['starting', 'training', 'preparing']:
            return jsonify({'error': '실행 중인 작업은 삭제할 수 없습니다'}), 400

        # Job 삭제
        del training_jobs[job_id]

        # 관련 파일 삭제 (모델 파일, 로그 파일 등)
        # cleanup_job_files(job_id)

        return jsonify({
            'success': True,
            'message': f'Job {job_id}가 삭제되었습니다'
        })

    except Exception as e:
        logger.error(f"Job 삭제 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Job 취소"""
    try:
        if job_id not in training_jobs:
            return jsonify({'error': '작업을 찾을 수 없습니다'}), 404

        job_data = training_jobs[job_id]
        status = job_data.get('status')

        # 취소 가능한 상태 확인
        if status not in ['starting', 'training', 'preparing']:
            return jsonify({'error': f'상태가 {status}인 작업은 취소할 수 없습니다'}), 400

        # 실제 구현에서는 WebFineTuner 인스턴스에 취소 신호 전송
        # finetuner = get_finetuner_instance(job_id)
        # if finetuner:
        #     finetuner.cancel_training()

        # 상태 업데이트
        training_jobs[job_id]['status'] = 'cancelled'
        training_jobs[job_id]['end_time'] = datetime.now().isoformat()

        return jsonify({
            'success': True,
            'message': f'Job {job_id}가 취소되었습니다'
        })

    except Exception as e:
        logger.error(f"Job 취소 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-model/<job_id>', methods=['GET'])
def download_model(job_id):
    """훈련된 모델 다운로드"""
    try:
        job = training_jobs.get(job_id)
        if not job or job['status'] != 'completed':
            return jsonify({'error': '완료된 모델이 없습니다'}), 404

        model_path = job['result']['model_path']

        if not os.path.exists(model_path):
            return jsonify({'error': '모델 파일을 찾을 수 없습니다'}), 404

        # 모델 파일들을 ZIP으로 압축
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, model_path)
                    zf.write(file_path, arc_name)

        memory_file.seek(0)

        return send_file(
            io.BytesIO(memory_file.read()),
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'model_{job_id}.zip'
        )

    except Exception as e:
        logger.error(f"모델 다운로드 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test-generation', methods=['POST'])
def test_generation():
    """모델 생성 테스트"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다'}), 400

        job_id = data.get('job_id')
        prompt = data.get('prompt')

        if not job_id or not prompt:
            return jsonify({'error': '필수 필드 누락'}), 400

        job = training_jobs.get(job_id)
        if not job or job['status'] != 'completed':
            return jsonify({'error': '완료된 모델이 없습니다'}), 404

        # 간단한 생성 테스트 (실제로는 모델을 로드해서 생성)
        response = f"[모델 응답] {prompt}에 대한 답변입니다."

        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        logger.error(f"생성 테스트 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 에러 핸들러
@app.errorhandler(413)
def too_large(e):
    return jsonify(success=False, error="파일 크기가 너무 큽니다. (최대 16MB)"), 413


@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"내부 서버 오류: {str(e)}")
    return jsonify(success=False, error="내부 서버 오류가 발생했습니다."), 500


# WebSocket 이벤트
@socketio.on('connect')
def handle_connect():
    logger.info('클라이언트 연결됨')


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('클라이언트 연결 해제됨')


if __name__ == '__main__':
    # 필요한 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)

    logger.info("서버 시작 중...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)