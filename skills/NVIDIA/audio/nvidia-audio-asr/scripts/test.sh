mkdir -p /workspace/logs

python3 -u /workspace/infer_runner.py \
  --model_id sensevoice-small \
  --model_dir /data/models/speech_recognition/SenseVoiceSmall \
  --output_dir /workspace/results/predictions/ \
  --acc_report /workspace/results/acc_report.json \
  --data_set /workspace/config/data_set.cfg 2>&1 | tee /workspace/logs/test.log