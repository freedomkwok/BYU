mkdir BYU
cd BYU
apt update
apt install unzip
pip install gdown
pip install tmux

find ./yolo_dataset/ -name "*.jpg" -delete

unzip -j yolo_dataset_009.zip 'yolo_dataset_009/*' -d yolo_dataset
unzip yolo_dataset.zip 'yolo_dataset/*' -d ./

unzip yolo_dataset_009.zip 'yolo_dataset_009/*' -d yolo_dataset

gdown --folder https://drive.google.com/drive/folders/1zEjBVsjzyGKIhyw4HVnUSUFByf_FNeW6 --fuzzy --quiet

gdown 'https://drive.google.com/file/d/1nPxH3jm9LXEI5HPx-5d9K8OwsFW5qXQQ/view?usp=sharing' --fuzzy #cmd.py
gdown 'https://drive.google.com/file/d/1f8JZn4LM9kvb_WEFB53l5_VD0rqwrjg5/view?usp=sharing' --fuzzy #ipynb.py
gdown 'https://drive.google.com/file/d/1v7cqVi4D2Q8qFRHWdoKvLAvaQlFIAv0D/view?usp=sharing' --fuzzy  #trainer.py
gdown 'https://drive.google.com/file/d/1RQ0W6gipuz_P3UkIIkePNQp13HdFhJC3/view?usp=sharing' --fuzzy  #move.py


import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIA5QRIRGY3KJAP56V7',
    aws_secret_access_key='oVEavPXCXJFgbLoNjPFQIvrfd/m4CzmfXTLbVNpa',
    region_name='us-west-2'  # Optional: set your region
)

bucket_name = "runpod20250528"
local_file = "/workspace/BYU/notebooks/yolo_weights/motor_detector_shared_009_full_optuna_trial_92.zip "
s3_key = "motor_detector_shared_009_full_optuna_trial_92.zip"

s3.upload_file(local_file, bucket_name, s3_key)



scp root@5c5ea32730fb:/workspace/BYU/notebooks/yolo_weights/motor_detector_shared_009_full_optuna_trial_92.zip ./motor_detector_shared_009_full_optuna_trial_92.zip
scp -P 16107 -i ~/.ssh/id_ed25519 root@209.170.80.132:/workspace/BYU/notebooks/yolo_weights/motor_detector_shared_009_full_optuna_trial_92.zip ~/motor_detector_shared_009_full_optuna_trial_92.zip

ps -eo pid,ppid,state,comm | awk '$3 == "Z" && $4 == "pt_main_thread" { print $2 }' | sort -u | xargs -r kill -9

ps -eo pid,ppid,state,cmd | awk '/python trainer.py --dataset shared_010_scaled --study shared_010_scaled --custom_model b5 --f_epoch 2/ { print $1 }' | xargs -r kill -9


mkdir notebooks
mv ./trainer.py ./notebooks/
mv ./requirements.txt ./notebooks/
mv ./local-eda-visualization-yolov8.ipynb ./notebooks/
unzip ./yolo_dataset.zip -d ./notebooks/

cd notebooks
unzip
pip install optuna
pip install -e ../ultralytics
pip install -r requirements.txt

tmux new -s train
tmux attach -t train
tmux kill-session -t train

zip -r /workspace/BYU/notebooks/yolo_dataset/motor_detector_shared_optuna_trial_21.zip /workspace/BYU/notebooks/yolo_dataset/motor_detector_shared_optuna_trial_21

apt update
apt install nano
apt install zip
apt install unzip
pip install gdown
pip install tmux
pip install -r requirements.txt

apt update
apt install nano
apt install zip
apt install unzip
pip install gdown
pip install tmux
pip install -r requirements.txt



I have 10 years of experience in full-stack, backend, and big data technologies, having worked at four different ad-tech companies. I’ve mastered four programming languages—Scala, Python, Go, and C#—along with their associated ecosystems to achieve technical and business goals.
Recent Projects:
	•	Participated in Kaggle machine learning and LLM competitions such as BYU - Locating Bacterial Flagellar Motors 2025 and WSDM Cup - Multilingual Chatbot Arena, contributing to data engineering tasks including preprocessing, post-processing, and setting up ML pipelines.
	•	Developed MLOps workflows to train multiple models with different parameters efficiently, and created tools to visualize results and performance metrics.
Company Projects:
	•	Built Spotify’s 500M audience targeting system from scratch.
	•	Designed and implemented Samsung TV’s sessionizer and channel program pipeline, along with the supporting architecture.