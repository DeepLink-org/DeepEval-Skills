mkdir logs

model_path="/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"

python3 -m sglang.launch_server \
        --model ${model_path} \
        --tp 8 \
        --trust-remote-code \
        --port 30000 2>&1|tee ./logs/serve.log
  

