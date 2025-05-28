import os
import shutil
import argparse

# Define file paths

local_dev =  "/workspace/BYU/notebooks" if "WANDB_API_KEY" in os.environ else "C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks"
google_folder = r"G:\My Drive\BYU\notebooks"

file_pairs = [
    (
        os.path.join(local_dev, 'trainer.py'),
        os.path.join(google_folder, 'trainer.py')
    ),
    (
        os.path.join(local_dev, 'local-eda-visualization-yolov8.ipynb'),
        os.path.join(google_folder, 'local-eda-visualization-yolov8.ipynb')
    ),
    (
        os.path.join(local_dev, 'requirements.txt'),
        os.path.join(google_folder, 'requirements.txt')
    ),
    (
        os.path.join(local_dev, 'cmd'),
        os.path.join(google_folder, 'cmd')
    ),
    (
        os.path.join(local_dev, './utils/move_script.py'),
        os.path.join(google_folder, 'move_script.py')
    ),
    (
        os.path.join(local_dev, './utils/move_script2.py'),
        os.path.join(google_folder, 'move_script2.py')
    ),
    (
        os.path.join(local_dev, './utils/get_results.py'),
        os.path.join(google_folder, 'get_results.py')
    ),
]

def sync_files(reverse=False):
    for src, dst in file_pairs:
        real_src, real_dst = (dst, src) if reverse else (src, dst)
        try:
            if not os.path.exists(real_dst) or os.path.getmtime(real_src) > os.path.getmtime(real_dst):
                os.makedirs(os.path.dirname(real_dst), exist_ok=True)
                shutil.copy2(real_src, real_dst)
                print(f"Synced: {real_src} -> {real_dst}")
            else:
                print(f"Skipped (up-to-date): {real_src}")
        except Exception as e:
            print(f"Error syncing {real_src}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync files between local and Google Drive folders.")
    parser.add_argument('--reverse', action='store_true', help='Sync from Google Drive to local instead of local to Drive')
    args = parser.parse_args()

    sync_files(reverse=args.reverse)
