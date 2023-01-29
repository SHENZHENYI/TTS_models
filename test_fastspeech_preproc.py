from src.utils.preprocessor.fastspeech_preproc import Preprocessor
from config.fastspeech2.preprocess import PreprocessConfig

cfg = PreprocessConfig()
preproc = Preprocessor(cfg)

preproc.build_from_path()
