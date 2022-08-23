import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

import fire, os, json
import logging
from modules.utils import create_logger

from modules.io.minio_utils import *
from modules.datasets import *
from modules.models.trainning import train
from modules.xai.ACE_analysis import ace_analysis
from modules.xai.SPACE_analysis import space_analysis
from modules.xai.ECLAD_analysis import eclad_analysis
from modules.xai.ConceptShap_analysis import cshap_analysis
from modules.xai.model_eval import model_eval, models_eval
from modules.xai.associate_concepts import associate_CE
from modules.xai.scatter_report import scatterplot_report_CE

def test_task(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

def task(config_path):
    logger = create_logger("TASK")
    for k,v in os.environ.items():
        if "SLURM" in k: logger.info(f"{k}: {v}")
    with open(config_path, "r") as f:
        conf = json.load(f)
    if conf["taskname"] in globals():
        globals()[conf["taskname"]](*conf["args"], **conf["kwargs"])
    else:
        logger.info(f"TASK {conf['taskname']} does not exist")


if __name__ == '__main__':
    # create logger
    logger = create_logger(name="general", level = logging.DEBUG)

    # starting
    logger.info("STARTING")

    # launch process
    fire.Fire()

    logger.info("FINISHED")