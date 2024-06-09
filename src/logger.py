from datetime import datetime


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.log_file = "log.csv"
        # Create file if it doesn't exist
        with open(self.log_file, "a") as f:
            f.write("timestamp,model,data_set,epoch,loss,accuracy\n")

    def log(self, model, data_set, epoch='', loss='', accuracy=''):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp},{model},{data_set},{epoch},{loss},{accuracy}\n")
