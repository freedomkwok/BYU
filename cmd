mkdir BYU
cd BYU
apt update
apt install unzip
pip install gdown
pip install tmux


gdown --folder https://drive.google.com/drive/folders/1zEjBVsjzyGKIhyw4HVnUSUFByf_FNeW6 --fuzzy --quiet

gdown 'https://drive.google.com/file/d/1nPxH3jm9LXEI5HPx-5d9K8OwsFW5qXQQ/view?usp=sharing' --fuzzy #cmd.py
gdown 'https://drive.google.com/file/d/1f8JZn4LM9kvb_WEFB53l5_VD0rqwrjg5/view?usp=sharing' --fuzzy #ipynb.py
gdown 'https://drive.google.com/file/d/1v7cqVi4D2Q8qFRHWdoKvLAvaQlFIAv0D/view?usp=sharing' --fuzzy  #trainer.py
gdown 'https://drive.google.com/file/d/1RQ0W6gipuz_P3UkIIkePNQp13HdFhJC3/view?usp=sharing' --fuzzy  #move.py


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