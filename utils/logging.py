import logging

class Logger:
    def __init__(self, log_dir_path="", log_dir_name=""):

        logFormatter = logging.Formatter("%(asctime)s %(message)s")
        self.logger_obj = logging.getLogger()

        if log_dir_path != "" and log_dir_name != "":
            fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir_path, log_dir_name))
            fileHandler.setFormatter(logFormatter)
            self.logger_obj.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger_obj.addHandler(consoleHandler)