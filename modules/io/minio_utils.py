from pathlib import Path
from minio import Minio
import json, time, shutil
import fire
import logging
from modules.utils import create_logger

class MinioCustom(Minio):
    def fget(self, bucket_name, object_name, file_path):
        """
        This method will get the file or content from <object_name> (can be a folder), downloading it into the file <file_path>.
        the final path will be:
        ex1:
        file_path="./some_folder"
        object_name="some_folder_online/the_file.txt"
        result
        "./some_folder/the_file.txt"
        ex2:
        file_path="./some_folder"
        object_name="some_folder_online/the_folder"
        result
        "./some_folder/the_folder"
        """
        # list all items in bucket/folder
        objects_list = self.list_objects(
            bucket_name=bucket_name,
            prefix=object_name,
            recursive=True,
            include_version=False,
        )
        # get each item 
        for item in objects_list:
            item_path=(Path(file_path)/object_name).as_posix()+item.object_name.replace(object_name, "")
            
            self.fget_object(
                bucket_name=bucket_name,
                object_name=item.object_name,
                file_path=item_path, 
                request_headers=None,
                ssec=None,
                version_id=None,
                extra_query_params=None,
            ) 
    def fput(self, bucket_name, object_name, file_path):
        """
        This method will put the file or content from <file_path> (can be a folder), uploading it into the object <object_name>.
        the final path will be:
        ex1:
        file_path="./some_folder/the_file.txt"
        object_name="some_folder_online"
        result object
        "some_folder_online/the_file.txt"
        ex2:
        file_path="./some_folder/the_folder"
        object_name="some_folder_online"
        result object
        "some_folder_online/the_folder"
        """
        # list all items in bucket/folder
        if Path(file_path).is_file():
            objects_list = [Path(file_path).as_posix()]
        else:
            objects_list = [p.as_posix() for p in Path(file_path).glob("**/*") if p.is_file()]

        # get each item 
        for item in objects_list:
            item_path=item.replace(Path(file_path).parent.as_posix(), "").lstrip("/")
            if len(object_name)>0: item_path = object_name + "/" + item_path
            self.fput_object(
                bucket_name=bucket_name,
                object_name=item_path,
                file_path=item,
                content_type='application/octet-stream',
                part_size=5242880
            )
    def fdel(self, bucket_name, object_name):
        """
        This method will delete the file or content from <object_name>.
        """
        # list all items in bucket/folder
        objects_list = self.list_objects(
            bucket_name=bucket_name,
            prefix=object_name,
            recursive=True,
            include_version=False,
        )
        # delete each item 
        for item in objects_list:
            item_path=(Path(file_path)/object_name).as_posix()+item.object_name.replace(object_name, "")
            self.remove_object(
                bucket_name=bucket_name, 
                object_name=item.object_name
            )

def sync_folder_down(path_minio_config="./data/secrets/minio.json", folder="./data/datasets/AB", bucket="cedatasets", force=False):
    logger = create_logger(name="minio", level = logging.DEBUG)
    # read config
    with open(path_minio_config, "r") as f:
        minio_config = json.load(f)
    # create minio client
    minio_client = MinioCustom(**minio_config)
    logger.debug("minio client created")
    # check if exitsts locally
    if Path(folder).exists():
        if force==True:
            shutil.rmtree(folder)
        else:
            return False
    # check if exists online
    existing_objects = [n.object_name for n in minio_client.list_objects(bucket)]
    if not ((Path(folder).name+".zip") in existing_objects):
        return False
    # try to download
    for i in range(5): # 5 retries
        try:
            minio_client.fget(bucket, Path(folder).name+".zip", Path(folder).parent.as_posix())
            break
        except:
            time.sleep(10)
    # unzip
    logger.debug("unzipping dataset")
    shutil.unpack_archive(folder+".zip", Path(folder).parent.as_posix(), format="zip")
    # remove zip
    Path(folder+".zip").unlink()
    logger.debug("dataset downloaded")
    return True

def sync_folder_up(path_minio_config="./data/secrets/minio.json", bucket="datasets", folder="./data/datasets/AB", force=False):
    logger = create_logger(name="minio", level = logging.DEBUG)
    # read config
    with open(path_minio_config, "r") as f:
        minio_config = json.load(f)
    # create minio client
    minio_client = MinioCustom(**minio_config)
    logger.debug("minio client created")
    # check if exitsts locally
    if not Path(folder).exists():
        return False
    # check if exists online
    existing_objects = [n.object_name for n in minio_client.list_objects(bucket)]
    if ((Path(folder).name+".zip") in existing_objects) and (not force):
        return False
    # assemble paths
    base_name = folder
    root_dir = Path(folder).parent.as_posix()
    base_dir = Path(folder).name
    zip_file = Path(folder+".zip").as_posix()
    # zip
    logger.debug("zipping")
    shutil.make_archive(
      base_name=base_name, 
      format="zip", 
      root_dir=root_dir, 
      base_dir=base_dir
    )
    # upload
    for i in range(5): # 5 retries
        try:
            minio_client.fput(bucket, "", zip_file)
            logger.debug("uploaded.")
            Path(zip_file).unlink()
            break
        except:
            logger.debug("error uploading")
            time.sleep(10)
    return True

def sync_folders_down(path_minio_config="./data/secrets/minio.json", folder="./data/datasets", bucket="cedatasets", force=False, filter=[]):
    logger = create_logger(name="minio", level = logging.DEBUG)
    # read config
    with open(path_minio_config, "r") as f:
        minio_config = json.load(f)
    # create minio client
    minio_client = MinioCustom(**minio_config)
    logger.info("minio client created.")
    # get items from bucket
    existing_objects = sorted([n.object_name for n in minio_client.list_objects(bucket)])
    # download each item
    for item in existing_objects:
        if (len(filter)>0) and (not any([f in item for f in filter])): continue
        logger.info(f"downloading: {item}")
        # check if exitsts locally
        if (Path(folder)/(item.replace(".zip", ""))).exists():
            if force==True:
                logger.info(" item already exists... using force.")
                shutil.rmtree((Path(folder)/(item.replace(".zip", ""))).as_posix())
            else:
                logger.info(" item already exists.")
                continue
        # try to download
        for i in range(5): # 5 retries
            try:
                minio_client.fget(bucket, item, Path(folder).as_posix())
                break
            except:
                time.sleep(10)
        # unzip
        logger.info("unzipping dataset.")
        shutil.unpack_archive((Path(folder)/item).as_posix(), Path(folder).as_posix(), format="zip")
        # remove zip
        (Path(folder)/item).unlink()
        logger.info("dataset downloaded.")
    logger.info("finished")
    return True

def sync_folders_up(path_minio_config="./data/secrets/minio.json", bucket="cedatasets", folder="./data/datasets", force=False, filter=[]):
    logger = create_logger(name="minio", level = logging.DEBUG)
    # read config
    with open(path_minio_config, "r") as f:
        minio_config = json.load(f)
    # create minio client
    minio_client = MinioCustom(**minio_config)
    logger.info("minio client created.")
    # check if exitsts locally
    if not Path(folder).exists():
        logger.info("folder does not exists locally.")
        return False
    # download each forlder
    for item_path in sorted(list(Path(folder).glob("*"))):
        if (len(filter)>0) and (not any([f in item_path.name for f in filter])): continue
        if not item_path.is_dir(): continue
        logger.info(f"uploading: {item_path.name}")
        # check if exists online
        existing_objects = [n.object_name for n in minio_client.list_objects(bucket)]
        if ((Path(item_path).name+".zip") in existing_objects) and (not force):
            logger.info(" item already exists.")
            continue
        # assemble paths
        base_name = item_path.as_posix()
        root_dir = Path(item_path).parent.as_posix()
        base_dir = Path(item_path).name
        zip_file = Path(item_path.as_posix()+".zip").as_posix()
        # zip
        logger.info("zipping.")
        shutil.make_archive(
          base_name=base_name, 
          format="zip", 
          root_dir=root_dir, 
          base_dir=base_dir
        )
        # upload
        for i in range(5): # 5 retries
            try:
                minio_client.fput(bucket, "", zip_file)
                logger.info("uploaded.")
                Path(zip_file).unlink()
                break
            except Exception as e:
                logger.info(f"error uploading: {e}")
                time.sleep(10)
    logger.info("finished.")
    return True


