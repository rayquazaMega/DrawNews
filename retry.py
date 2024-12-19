import functools
import logging
import time


class Retry:
    """
    处理异常的上下文管理器 + 迭代器 + 装饰器
    ```
    import random

    # 1. 作为装饰器使用
    retry = Retry(max_tries=3, delay=1)

    @retry
    def unstable_operation():
        '''模拟一个不稳定的操作'''
        if random.random() < 0.7:  # 70% 概率失败
            raise ValueError("Random failure occurred")
        return "Operation successful"

    # 装饰器使用示例
    try:
        result = unstable_operation()
        print(f"Decorator result: {result}")
    except Exception as e:
        print(f"Final failure: {e}")

    # 2. 作为上下文管理器使用
    def risky_operation():
        if random.random() < 0.7:
            raise RuntimeError("Something went wrong")
        return "Success"

    retry_context = Retry(max_tries=3, delay=1)
    with retry_context:
        result = risky_operation()
        print(f"Context manager result: {result}")

    # 3. 作为迭代器使用
    def another_risky_operation():
        if random.random() < 0.7:
            raise ValueError("Operation failed")
        return "Success"

    retry_iterator = Retry(max_tries=3, delay=1)
    for attempt in retry_iterator:
        with attempt:
            result = another_risky_operation()
            print(f"Iterator result: {result}")

    # 4. 组合使用示例
    @Retry(max_tries=3, delay=1)
    def combined_operation():
        retry_inner = Retry(max_tries=2, delay=1)
        with retry_inner:
            if random.random() < 0.7:
                raise ValueError("Inner operation failed")
            return "Combined operation successful"

    try:
        result = combined_operation()
        print(f"Combined result: {result}")
    except Exception as e:
        print(f"Combined operation failed: {e}")
    ```
    """

    def __init__(self, max_tries=5, delay=2, logger=None):
        self.max_tries = max_tries
        self.delay = delay
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def __iter__(self):
        for i in range(self.max_tries):
            yield self
            if self.allright or i == self.max_tries - 1:
                if self.allright:
                    self.logger.debug("Operation succeeded, stopping iterations")
                else:
                    self.logger.warning("Max retries reached")
                return
            self.logger.info(
                f"Attempt {i+1} failed, waiting {self.delay} seconds before next try"
            )
            time.sleep(self.delay)

    def __enter__(self):
        self.logger.debug("Entering context manager")
        self.allright = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.allright = True
            self.logger.debug("Context exited successfully")
        else:
            self.logger.error(f"Exception occurred: {exc_val}")
            print(exc_val)
        return True

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_tries):
                try:
                    result = func(*args, **kwargs)
                    self.logger.debug(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                    )
                    return result
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_tries - 1:
                        self.logger.info(
                            f"Waiting {self.delay} seconds before next attempt"
                        )
                        time.sleep(self.delay)
                    else:
                        self.logger.error(
                            f"Max retries ({self.max_tries}) reached for {func.__name__}"
                        )
                        raise

        return wrapper
