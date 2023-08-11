from sensor.logger import logging
from sensor.exception import SensorException

def test_logger_and_exception():
     try:
          logging.info("Starting the test_logger_and_exception")
          result = 3/0
          print(result)
          logger.info("Stopping the test_logger_and_exception")
     except Exception as e:
          logger.info("Stopping the test_logger_and_exception")
          
          raise SensorException(e, sys) 


if __name__=="__main__":
     try:
          test_logger_and_exception()
     except Exception as e:
          print(e)