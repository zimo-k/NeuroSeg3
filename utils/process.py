import os
import datetime

def make_log_dir(root_dir, model_name, batch_size):
    start_time = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    checkpoint_dir = os.path.join(root_dir, f'{model_name}_{batch_size}_{str(start_time)}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def cal_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"It takes {hours}hours, {minutes}minutes, {seconds}seconds to finish training.")