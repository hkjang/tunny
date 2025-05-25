import React, { useState, useEffect, useRef } from 'react';
import { Upload, Play, Download, Settings, Database, Brain, Zap, CheckCircle, AlertCircle, Clock, RefreshCw, List, Eye, Search } from 'lucide-react';

const FineTuningWebApp = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedModel, setSelectedModel] = useState('');
  const [taskType, setTaskType] = useState('generation');
  const [trainingData, setTrainingData] = useState(null);
  const [validationData, setValidationData] = useState(null);
  const [trainingConfig, setTrainingConfig] = useState({
    num_epochs: 3,
    batch_size: 4,
    learning_rate: 5e-5,
    max_length: 512,
    warmup_steps: 100,
    weight_decay: 0.01,
    gradient_checkpointing: true,
    logging_steps: 50,
    eval_steps: 100,
    save_steps: 200
  });
  const [availableModels, setAvailableModels] = useState({});
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [progressLog, setProgressLog] = useState([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [statusCheckInterval] = useState(null);
  const intervalRef = useRef(null);

    // Job 상태 조회 관련 상태
  const [showJobStatus, setShowJobStatus] = useState(false);
  const [searchJobId, setSearchJobId] = useState('');
  const [jobStatusData, setJobStatusData] = useState(null);
  const [jobStatusLoading, setJobStatusLoading] = useState(false);
  const [jobStatusError, setJobStatusError] = useState(null);
  const [allJobs, setAllJobs] = useState([]);
  const [allJobsLoading, setAllJobsLoading] = useState(false);

  // API 기본 URL
  const API_BASE_URL = 'http://localhost:5000';

  useEffect(() => {
    loadAvailableModels();
  }, []);

  // 훈련 상태 체크를 위한 useEffect
  useEffect(() => {
    if (currentJobId && trainingStatus === 'training') {
      if (!intervalRef.current) {
        const interval = setInterval(checkTrainingStatus, 2000);
        intervalRef.current = interval;
      }
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [currentJobId, trainingStatus]);

  // 컴포넌트 언마운트 시 인터벌 정리
  useEffect(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
      }
    };
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setAvailableModels(data);
      const firstModel = Object.keys(data)[0];
      if (firstModel) setSelectedModel(firstModel);
    } catch (err) {
      console.error('모델 목록 로딩 실패:', err);
      setUploadError('모델 목록을 불러올 수 없습니다. 서버가 실행 중인지 확인하세요.');
    }
  };

    // 모든 Job 목록 조회
  const loadAllJobs = async () => {
    setAllJobsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/jobs?sort_by=created_at&sort_order=desc`);
      if (!response.ok) throw new Error('Job 목록 조회 실패');

      const data = await response.json();
      setAllJobs(data.jobs || []);
    } catch (error) {
      console.error('Job 목록 조회 오류:', error);
    } finally {
      setAllJobsLoading(false);
    }
  };

  // 특정 Job ID로 상태 조회
  const searchJobStatus = async (jobId) => {
    if (!jobId.trim()) {
      setJobStatusError('Job ID를 입력해주세요.');
      return;
    }

    setJobStatusLoading(true);
    setJobStatusError(null);
    setJobStatusData(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/training-status/${jobId.trim()}`);
      if (!response.ok) {
        throw new Error('Job을 찾을 수 없습니다.');
      }

      const data = await response.json();
      setJobStatusData({
        job_id: jobId.trim(),
        ...data
      });
    } catch (error) {
      setJobStatusError(error.message);
    } finally {
      setJobStatusLoading(false);
    }
  };

  // 훈련 상태 체크 함수
  const checkTrainingStatus = async () => {
  if (!currentJobId) return;

  try {
    const response = await fetch(`${API_BASE_URL}/api/training-status/${currentJobId}`);
    if (!response.ok) throw new Error('상태 체크 실패');

    const statusData = await response.json();

    setTrainingStatus(statusData.status);

    if (typeof statusData.progress === 'number') {
      setTrainingProgress(statusData.progress);
    }

    if (Array.isArray(statusData.logs)) {
      setProgressLog(prev => {
        const existingLogCount = prev.length;
        const newLogs = statusData.logs.slice(existingLogCount);
        return [...prev, ...newLogs];
      });
    }

    if (statusData.status === 'completed' || statusData.status === 'failed') {
      setTrainingProgress(statusData.status === 'completed' ? 100 : trainingProgress);
      setProgressLog(prev => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          type: statusData.status === 'completed' ? 'success' : 'error',
          message:
            statusData.status === 'completed'
              ? '훈련이 성공적으로 완료되었습니다!'
              : (statusData.error || '훈련 중 오류가 발생했습니다.')
        }
      ]);
    }
  } catch (error) {
    setProgressLog(prev => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        type: 'warning',
        message: '상태 체크 중 오류 발생 (훈련은 계속 진행 중)'
      }
    ]);
  }
};

  const handleFileUpload = async (file, type) => {
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);

    // 파일 크기 체크 (16MB)
    if (file.size > 16 * 1024 * 1024) {
      setUploadError('파일 크기가 너무 큽니다. (최대 16MB)');
      setIsUploading(false);
      return;
    }

    // 파일 형식 체크
    const allowedTypes = ['application/json', 'text/csv', 'text/plain'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['json', 'csv', 'txt'];

    if (!allowedExtensions.includes(fileExtension)) {
      setUploadError('지원하지 않는 파일 형식입니다. (JSON, CSV 파일만 지원)');
      setIsUploading(false);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload-data`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok && result.success) {
        const data = {
          data: result.data,
          total_rows: result.data.length,
          filename: result.filename || file.name,
          columns: ["input", "output"],
        };

        if (type === 'training') {
          setTrainingData(data);
        } else {
          setValidationData(data);
        }

        setUploadError(null);
      } else {
        throw new Error(result.error || '파일 업로드에 실패했습니다.');
      }
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      setUploadError(error.message || '파일 업로드 중 오류가 발생했습니다.');
    } finally {
      setIsUploading(false);
    }
  };

  const startTraining = async () => {
    if (!trainingData) {
      alert('훈련 데이터를 업로드해주세요.');
      return;
    }

    if (!selectedModel) {
      alert('모델을 선택해주세요.');
      return;
    }

    const payload = {
      model_name: selectedModel,
      task_type: taskType,
      train_data: trainingData.data,
      val_data: validationData?.data || [],
      ...trainingConfig,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/start-training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        setCurrentJobId(result.job_id);
        setTrainingStatus('training');
        setCurrentStep(4);
        setProgressLog([{
          timestamp: new Date().toLocaleTimeString(),
          type: 'info',
          message: '훈련이 시작되었습니다...'
        }]);
        setTrainingProgress(0);
      } else {
        throw new Error(result.error || '훈련 시작에 실패했습니다.');
      }
    } catch (error) {
      alert('훈련 시작 실패: ' + error.message);
    }
  };

  const downloadModel = async () => {
    if (!currentJobId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/download-model/${currentJobId}`);

      if (!response.ok) {
        throw new Error('모델 다운로드 실패');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${currentJobId}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('모델 다운로드 오류:', error);
      alert('모델 다운로드 실패: ' + error.message);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'training':
      case 'running':
        return <Clock className="w-5 h-5 text-orange-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'training':
      case 'running':
        return '훈련 중';
      case 'completed':
        return '완료됨';
      case 'failed':
      case 'error':
        return '실패';
      case 'pending':
        return '대기 중';
      default:
        return status || '알 수 없음';
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '-';
    return new Date(timestamp).toLocaleString('ko-KR');
  };

  const renderJobStatusView = () => (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Eye className="w-6 h-6 text-blue-600 mr-3" />
          <h2 className="text-2xl font-bold text-gray-800">Job 상태 조회</h2>
        </div>
        <button
          onClick={() => setShowJobStatus(false)}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          훈련으로 돌아가기
        </button>
      </div>

      {/* Job ID 검색 */}
      <div className="mb-8">
        <div className="flex gap-4 mb-4">
          <div className="flex-1">
            <input
              type="text"
              value={searchJobId}
              onChange={(e) => setSearchJobId(e.target.value)}
              placeholder="Job ID를 입력하세요 (예: job_12345)"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              onKeyPress={(e) => e.key === 'Enter' && searchJobStatus(searchJobId)}
            />
          </div>
          <button
            onClick={() => searchJobStatus(searchJobId)}
            disabled={jobStatusLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center"
          >
            {jobStatusLoading ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Search className="w-5 h-5" />
            )}
          </button>
        </div>

        {jobStatusError && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg mb-4">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
              <span className="text-red-700">{jobStatusError}</span>
            </div>
          </div>
        )}

        {jobStatusData && (
          <div className="bg-gray-50 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Job 상세 정보</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span className="text-sm font-medium text-gray-600">Job ID:</span>
                <p className="text-gray-800 font-mono">{jobStatusData.job_id}</p>
              </div>
              <div className="flex items-center">
                <span className="text-sm font-medium text-gray-600 mr-2">상태:</span>
                {getStatusIcon(jobStatusData.status)}
                <span className="ml-2 font-medium">{getStatusText(jobStatusData.status)}</span>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-600">진행률:</span>
                <div className="flex items-center mt-1">
                  <div className="flex-1 bg-gray-200 rounded-full h-2 mr-3">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${jobStatusData.progress || 0}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium">{(jobStatusData.progress || 0).toFixed(1)}%</span>
                </div>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-600">시작 시간:</span>
                <p className="text-gray-800">{formatTimestamp(jobStatusData.start_time)}</p>
              </div>
              {jobStatusData.end_time && (
                <div>
                  <span className="text-sm font-medium text-gray-600">종료 시간:</span>
                  <p className="text-gray-800">{formatTimestamp(jobStatusData.end_time)}</p>
                </div>
              )}
              {jobStatusData.model_name && (
                <div>
                  <span className="text-sm font-medium text-gray-600">모델:</span>
                  <p className="text-gray-800">{jobStatusData.model_name}</p>
                </div>
              )}
            </div>

            {jobStatusData.logs && jobStatusData.logs.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold text-gray-800 mb-2">최근 로그</h4>
                <div className="bg-white rounded border max-h-32 overflow-y-auto p-3">
                  {jobStatusData.logs.slice(-10).map((log, index) => (
                    <div key={index} className="text-sm text-gray-700 mb-1">
                      {typeof log === 'object' ? log.message : log}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {jobStatusData.status === 'completed' && (
              <div className="mt-4">
                <button
                  onClick={() => downloadModel(jobStatusData.job_id)}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center"
                >
                  <Download className="w-4 h-4 mr-2" />
                  모델 다운로드
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 전체 Job 목록 */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center">
            <List className="w-5 h-5 mr-2" />
            전체 Job 목록
          </h3>
          <button
            onClick={loadAllJobs}
            disabled={allJobsLoading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 transition-colors flex items-center"
          >
            {allJobsLoading ? (
              <RefreshCw className="w-4 h-4 animate-spin mr-2" />
            ) : (
              <RefreshCw className="w-4 h-4 mr-2" />
            )}
            새로고침
          </button>
        </div>

        <div className="bg-gray-50 rounded-lg overflow-hidden">
          {allJobsLoading ? (
            <div className="p-8 text-center">
              <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-gray-400" />
              <p className="text-gray-600">Job 목록을 불러오는 중...</p>
            </div>
          ) : allJobs.length === 0 ? (
            <div className="p-8 text-center text-gray-600">
              등록된 Job이 없습니다.
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {allJobs.map((job, index) => (
                <div key={index} className="p-4 hover:bg-white transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <span className="font-mono text-sm bg-gray-200 px-2 py-1 rounded mr-3">
                          {job.job_id}
                        </span>
                        <div className="flex items-center">
                          {getStatusIcon(job.status)}
                          <span className="ml-2 font-medium">{getStatusText(job.status)}</span>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600">
                        <div>
                          <span className="font-medium">모델:</span> {job.model_name || 'N/A'}
                        </div>
                        <div>
                          <span className="font-medium">진행률:</span> {(job.progress || 0).toFixed(1)}%
                        </div>
                        <div>
                          <span className="font-medium">시작:</span> {formatTimestamp(job.start_time)}
                        </div>
                        <div>
                          <span className="font-medium">종료:</span> {formatTimestamp(job.end_time)}
                        </div>
                      </div>
                    </div>
                    <div className="ml-4 flex gap-2">
                      <button
                        onClick={() => {
                          setSearchJobId(job.job_id);
                          searchJobStatus(job.job_id);
                        }}
                        className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
                      >
                        상세보기
                      </button>
                      {job.status === 'completed' && (
                        <button
                          onClick={() => downloadModel(job.job_id)}
                          className="px-3 py-1 bg-green-500 text-white text-sm rounded hover:bg-green-600 transition-colors"
                        >
                          다운로드
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );


  const renderStepIndicator = () => (
    <div className="flex items-center justify-center mb-8">
      {[1, 2, 3, 4].map((step) => (
        <div key={step} className="flex items-center">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold ${
            currentStep >= step 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-600'
          }`}>
            {step}
          </div>
          {step < 4 && (
            <div className={`w-16 h-1 mx-2 ${
              currentStep > step ? 'bg-blue-600' : 'bg-gray-200'
            }`} />
          )}
        </div>
      ))}
    </div>
  );

  const renderModelSelection = () => (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center mb-6">
        <Brain className="w-6 h-6 text-blue-600 mr-3" />
        <h2 className="text-2xl font-bold text-gray-800">모델 선택</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            기본 모델
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {Object.entries(availableModels).map(([key, model]) => (
              <option key={key} value={key}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            작업 유형
          </label>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value)}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="generation">텍스트 생성</option>
            <option value="classification">텍스트 분류</option>
          </select>
        </div>
      </div>

      <div className="bg-blue-50 p-4 rounded-lg mb-6">
        <h3 className="font-semibold text-blue-800 mb-2">선택된 모델 정보</h3>
        <p className="text-blue-700">
          {availableModels[selectedModel]?.name} - {taskType === 'generation' ? '텍스트 생성' : '텍스트 분류'} 작업
        </p>
        <p className="text-blue-600 text-sm mt-1">
          {availableModels[selectedModel]?.description}
        </p>
      </div>

      <button
        onClick={() => setCurrentStep(2)}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-semibold"
      >
        다음 단계
      </button>
    </div>
  );

  const renderDataUpload = () => (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center mb-6">
        <Database className="w-6 h-6 text-green-600 mr-3" />
        <h2 className="text-2xl font-bold text-gray-800">데이터 업로드</h2>
      </div>

      {uploadError && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
            <span className="text-red-700">{uploadError}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-green-500 transition-colors">
          <div className="text-center">
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-700 mb-2">훈련 데이터</h3>
            <p className="text-gray-500 mb-4">CSV 또는 JSON 파일을 업로드하세요</p>
            <input
              type="file"
              accept=".csv,.json"
              onChange={(e) => handleFileUpload(e.target.files[0], 'training')}
              className="hidden"
              id="training-file"
              disabled={isUploading}
            />
            <label
              htmlFor="training-file"
              className={`${isUploading ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'} text-white px-4 py-2 rounded-lg cursor-pointer inline-block transition-colors`}
            >
              {isUploading ? '업로드 중...' : '파일 선택'}
            </label>
            {trainingData && (
              <div className="mt-4 p-3 bg-green-50 rounded-lg">
                <p className="text-green-800 font-semibold">
                  ✓ {trainingData.total_rows}개 행 업로드됨
                </p>
                <p className="text-green-600 text-sm">{trainingData.filename}</p>
              </div>
            )}
          </div>
        </div>

        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-blue-500 transition-colors">
          <div className="text-center">
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-700 mb-2">검증 데이터 (선택사항)</h3>
            <p className="text-gray-500 mb-4">모델 성능 평가용 데이터</p>
            <input
              type="file"
              accept=".csv,.json"
              onChange={(e) => handleFileUpload(e.target.files[0], 'validation')}
              className="hidden"
              id="validation-file"
              disabled={isUploading}
            />
            <label
              htmlFor="validation-file"
              className={`${isUploading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'} text-white px-4 py-2 rounded-lg cursor-pointer inline-block transition-colors`}
            >
              {isUploading ? '업로드 중...' : '파일 선택'}
            </label>
            {validationData && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-blue-800 font-semibold">
                  ✓ {validationData.total_rows}개 행 업로드됨
                </p>
                <p className="text-blue-600 text-sm">{validationData.filename}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex gap-4 mt-8">
        <button
          onClick={() => setCurrentStep(1)}
          className="flex-1 bg-gray-500 text-white py-3 px-6 rounded-lg hover:bg-gray-600 transition-colors font-semibold"
        >
          이전
        </button>
        <button
          onClick={() => setCurrentStep(3)}
          disabled={!trainingData || isUploading}
          className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold"
        >
          다음 단계
        </button>
      </div>
    </div>
  );

  const renderTrainingConfig = () => (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center mb-6">
        <Settings className="w-6 h-6 text-purple-600 mr-3" />
        <h2 className="text-2xl font-bold text-gray-800">훈련 설정</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            에포크 수
          </label>
          <input
            type="number"
            value={trainingConfig.num_epochs}
            onChange={(e) => setTrainingConfig({...trainingConfig, num_epochs: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            min="1"
            max="100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            배치 크기
          </label>
          <input
            type="number"
            value={trainingConfig.batch_size}
            onChange={(e) => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            min="1"
            max="64"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            학습률
          </label>
          <input
            type="number"
            value={trainingConfig.learning_rate}
            onChange={(e) => setTrainingConfig({...trainingConfig, learning_rate: parseFloat(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            step="0.00001"
            min="0.00001"
            max="0.01"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            최대 시퀀스 길이
          </label>
          <input
            type="number"
            value={trainingConfig.max_length}
            onChange={(e) => setTrainingConfig({...trainingConfig, max_length: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            min="64"
            max="2048"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            워밍업 스텝
          </label>
          <input
            type="number"
            value={trainingConfig.warmup_steps}
            onChange={(e) => setTrainingConfig({...trainingConfig, warmup_steps: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            min="0"
            max="1000"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Weight Decay
          </label>
          <input
            type="number"
            value={trainingConfig.weight_decay}
            onChange={(e) => setTrainingConfig({...trainingConfig, weight_decay: parseFloat(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            step="0.001"
            min="0"
            max="0.1"
          />
        </div>
      </div>

      <div className="mt-6">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={trainingConfig.gradient_checkpointing}
            onChange={(e) => setTrainingConfig({...trainingConfig, gradient_checkpointing: e.target.checked})}
            className="mr-2"
          />
          <span className="text-sm font-medium text-gray-700">그래디언트 체크포인팅 사용</span>
        </label>
      </div>

      <div className="flex gap-4 mt-8">
        <button
          onClick={() => setCurrentStep(2)}
          className="flex-1 bg-gray-500 text-white py-3 px-6 rounded-lg hover:bg-gray-600 transition-colors font-semibold"
        >
          이전
        </button>
        <button
          onClick={startTraining}
          className="flex-1 bg-purple-600 text-white py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors font-semibold flex items-center justify-center"
        >
          <Play className="w-5 h-5 mr-2" />
          훈련 시작
        </button>
      </div>
    </div>
  );

  const renderTrainingProgress = () => (
  <div className="bg-white rounded-xl shadow-lg p-8">
    <div className="flex items-center mb-6">
      <Zap className="w-6 h-6 text-orange-600 mr-3" />
      <h2 className="text-2xl font-bold text-gray-800">훈련 진행상황</h2>
    </div>

    <div className="mb-6">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">전체 진행률</span>
        <span className="text-sm font-medium text-gray-700">{trainingProgress.toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full transition-all duration-300"
          style={{ width: `${trainingProgress}%` }}
        ></div>
      </div>
    </div>

    <div className="mb-6">
      <div className="flex items-center mb-2">
        {trainingStatus === 'training' && <Clock className="w-5 h-5 text-orange-500 mr-2 animate-spin" />}
        {trainingStatus === 'completed' && <CheckCircle className="w-5 h-5 text-green-500 mr-2" />}
        {trainingStatus === 'failed' && <AlertCircle className="w-5 h-5 text-red-500 mr-2" />}
        <span className="font-medium">
          {trainingStatus === 'training' && '훈련 중...'}
          {trainingStatus === 'completed' && '훈련 완료!'}
          {trainingStatus === 'failed' && '훈련 실패'}
        </span>
        {currentJobId && (
          <span className="ml-2 text-xs text-gray-500">Job ID: {currentJobId}</span>
        )}
      </div>
    </div>

    <div className="bg-gray-50 rounded-lg p-4 mb-6 max-h-64 overflow-y-auto">
      <h3 className="font-semibold text-gray-800 mb-3">훈련 로그</h3>
      <div className="space-y-1">
        {progressLog.map((log, index) => (
          <div key={index} className="text-sm">
            <span className="text-gray-500">[{log.timestamp}]</span>{' '}
            <span className={
              log.type === 'success' ? 'text-green-600' :
              log.type === 'error' ? 'text-red-600' :
              log.type === 'warning' ? 'text-yellow-600' :
              'text-gray-700'
            }>
              {log.message}
            </span>
          </div>
        ))}
        {progressLog.length === 0 && (
          <p className="text-gray-500 text-sm">로그를 기다리는 중...</p>
        )}
      </div>
    </div>

    {/* 훈련 완료 시 버튼 */}
    {trainingStatus === 'completed' && (
      <div className="flex gap-4">
        <button
          onClick={downloadModel}
          className="flex-1 bg-green-600 text-white py-3 px-6 rounded-lg hover:bg-green-700 transition-colors font-semibold flex items-center justify-center"
        >
          <Download className="w-5 h-5 mr-2" />
          모델 다운로드
        </button>
        <button
          onClick={handleNewTraining}
          className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-semibold"
        >
          새 훈련 시작
        </button>
      </div>
    )}

    {/* 훈련 실패 시 버튼 */}
    {trainingStatus === 'failed' && (
      <div className="flex gap-4">
        <button
          onClick={handleRetryTraining}
          className="flex-1 bg-orange-600 text-white py-3 px-6 rounded-lg hover:bg-orange-700 transition-colors font-semibold"
        >
          다시 시도
        </button>
        <button
          onClick={handleNewTraining}
          className="flex-1 bg-gray-600 text-white py-3 px-6 rounded-lg hover:bg-gray-700 transition-colors font-semibold"
        >
          새 훈련 시작
        </button>
      </div>
    )}
  </div>
);

  // 헬퍼 함수들을 별도로 정의
  const handleNewTraining = () => {
    setCurrentStep(1);
    resetTrainingState();
  };

  const handleRetryTraining = () => {
    setCurrentStep(3);
    resetTrainingState();
  };

  const resetTrainingState = () => {
    setTrainingStatus(null);
    setProgressLog([]);
    setTrainingProgress(0);
    setCurrentJobId(null);

    // 인터벌 정리
    if (statusCheckInterval) {
      clearInterval(statusCheckInterval);
      useRef(null);
    }
  };

  return (
  <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">AI 모델 파인튜닝</h1>
        <p className="text-gray-600">간편한 웹 인터페이스로 AI 모델을 커스터마이징하세요</p>

        {/* Job 상태 조회 버튼 추가 */}
        <div className="mt-4">
          <button
            onClick={() => setShowJobStatus(true)}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center mx-auto"
          >
            <Eye className="w-5 h-5 mr-2" />
            Job 상태 조회
          </button>
        </div>
      </div>

      {/* showJobStatus 상태에 따라 Job 상태 조회 뷰 또는 메인 파인튜닝 프로세스 표시 */}
      {showJobStatus ? (
        renderJobStatusView()
      ) : (
        <>
          {renderStepIndicator()}

          {currentStep === 1 && renderModelSelection()}
          {currentStep === 2 && renderDataUpload()}
          {currentStep === 3 && renderTrainingConfig()}
          {currentStep === 4 && renderTrainingProgress()}
        </>
      )}
    </div>
  </div>
);



};

export default FineTuningWebApp;