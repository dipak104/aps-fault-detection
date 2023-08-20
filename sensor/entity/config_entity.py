import os, sys
from datetime import datetime

FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILENAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise SensorException(e,sys)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.database_name = "aps"
        self.collection_name = "sensor"
        self.data_inggestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
        self.feature_store_file_path = os.path.join(self.data_inggestion_dir, "feature_store", FILE_NAME)
        self.train_file_path = os.path.join(self.data_inggestion_dir, "dataset", TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.data_inggestion_dir, "dataset", TEST_FILE_NAME)
        self.test_size = 0.2
    
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)



class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
        self.missing_threshold:float = 0.7
        self.base_file_path = os.path.join("aps_failure_training_set1.csv")
    



class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transforation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transform_object_path = os.path.join(self.data_transforation_dir, "transformer", "transform.pkl")
        self.transformed_train_path = os.path.join(self.data_transforation_dir, "transformed", TRAIN_FILE_NAME)
        self.transformed_test_path = os.path.join(self.data_transforation_dir, "transformed", TEST_FILE_NAME)
        self.label_encoder_path = os.path.join(self.data_transforation_dir, "target_encoder", TARGET_ENCODER_OBJECT_FILE_NAME)
        self


class ModelTrainerConfig:...
class ModelEvaluationConfig:...
class ModelPusherConfig:...
