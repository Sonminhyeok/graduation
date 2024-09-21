import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 데이터 변경 감지 핸들러 클래스
class FileUpdateHandler(FileSystemEventHandler):
    def __init__(self, update_callback):
        self.update_callback = update_callback
    
    def on_modified(self, event):
        if event.src_path.endswith('.csv'):
            print(f"{event.src_path} 파일이 수정되었습니다. 모델을 다시 학습합니다.")
            self.update_callback()

# 파일 변경 감지 설정
def watch_for_updates(data_dir, update_callback):
    observer = Observer()
    event_handler = FileUpdateHandler(update_callback)
    observer.schedule(event_handler, path=data_dir, recursive=False)
    observer.start()
    print("파일 변경 감지 시작...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
