'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-12-12 09:38:51
* LastEditors: LiuFeng
* LastEditTime: 2023-12-12 09:39:11
* FilePath: /Acoustic_AD/TorchInversion/logger.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import logging
import os
from typing import Any

class _Logger:
    def __init__(self):
        self._loggers = {}
        self._active_logger = None
        self._log_dir = None

    def create_logger(self, name: str) -> logging.Logger:
        """Create a logger.

        Args:
            name (str): logger name.
            log_path (str): file path.

        Returns:
            logger.Logger: logger.
        """
        if name in self._loggers:
            raise ValueError(f"logger:'{name}' exists.")

        if self._log_dir is None:
            raise Exception("call `set_logdir` before creating logger.")

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
        file_handler = logging.FileHandler(os.path.join(self._log_dir, f"{name}.log"))
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(fmt)
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self._loggers[name] = logger
        self.__setattr__(name, logger)

        return logger

    def set_logdir(self, log_dir: str) -> None:
        """Set log directory for all loggers.

        Args:
            log_dir (str): directory.
        """
        assert self._log_dir is None and len(self._loggers) == 0

        if not os.path.exists(log_dir):
            try:
                os.makedirs(os.path.abspath(log_dir))
            except:
                pass

        self._log_dir = log_dir

    def set_logger(self, name: str) -> logging.Logger:
        """Set active logger.

        Args:
            name (str): logger name.
            
        Returns:
            logging.Logger: logger.
        """
        if name not in self._loggers:
            self.create_logger(name)
        self._active_logger = name
        return self._loggers[self._active_logger]
    

    def __getattribute__(self, __name: str) -> Any:
        try:
            return object.__getattribute__(self, __name)
        except:
            try:
                if self._active_logger is None:
                    raise NotImplementedError(
                        f"No logger available. Call `create_logger` or `set_logger` to initialize a logger."
                    )
                return getattr(self._loggers[self._active_logger], __name)
            except:
                raise

    def logdir(self) -> str:
        return self._log_dir

logger = _Logger()