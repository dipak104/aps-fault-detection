from sensor.logger import logging
from sensor.exception import SensorException

def test_logger_and_exception():
     try:
          result = 3/0
          print(result)
     except Exception as e:
          raise e


if __name__=="__main__":
     try:
          test_logger_and_exception()
     except Exception as e:
          print(e)