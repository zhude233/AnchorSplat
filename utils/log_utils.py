import logging
import os
import torch.distributed as dist

class ProcessSafeLogger:
    def __init__(self, log_file='output.log', log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_file = log_file
        
        # Check if this is the main process (rank 0)
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.propagate = False
    
    def get_logger(self):
        return self.logger
    

# Usage example:
if __name__ == "__main__":
    # Initialize the distributed environment here (for example, using torch.distributed.init_process_group)
    # dist.init_process_group(backend='nccl', init_method='env://')

    logger = ProcessSafeLogger().get_logger()
    logger.info("This log entry should appear only in the main process log file.")
