from pathlib import Path
import json
import random
import sys

sys.path.append(".")
from constants import RANDOM_SEED
import fire
import logging
from multiprocessing.dummy import Pool
import shutil
import itertools
import sys

random.seed(RANDOM_SEED)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
import shutil, errno


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def _copy_images(inputs):
    dest_root, src_file = inputs
    src_file = Path(src_file)
    dest_root = Path(dest_root)
    dest_root = dest_root.joinpath(src_file.parent.stem)
    dest_root.mkdir(exist_ok=True, parents=True)
    dest_file = dest_root.joinpath(f"{src_file.stem}.{src_file.suffix}")
    shutil.copy(src_file, dest_file)
    logger.debug(f"{src_file} -> {dest_file}")
    return True


def write_set(dest_root, which, resulted_dic, copy_images, pool):
    dest_root = Path(dest_root).joinpath(which)
    dest_root.mkdir(exist_ok=True, parents=True)
    dest_file = dest_root.joinpath("images.json")
    json.dump(resulted_dic, open(dest_file, "w"), indent=True)
    logger.info(f"training_set_recipe is written under {dest_file}")
    all_files = [item for sublist in list(resulted_dic.values()) for item in sublist]
    if copy_images:
        jobs = list(
                pool.map(
                        _copy_images,
                        zip(itertools.repeat(dest_root.joinpath("images")), all_files),
                )
        )
        try:
            assert sum(jobs) == len(all_files), "All files were not copied successfully"
        except AssertionError as exc:
            logger.error(exc)


class DatasetBuilder(object):
    _worker_pool = Pool()

    # @staticmethod
    # def make_asghar_set(train_json_path, dest_root, dest_root_to_images, val_set_counts=10):
    #     train_json_path = Path(train_json_path)
    #     dest_root = Path(dest_root)
    #     dest_root_to_images = Path(dest_root_to_images)
    #     all_train_images = json.load(open(train_json_path))
    #     for c in all_tra
    #
    #     pass


    @staticmethod
    def make_food_101_test_case(
            train_json_path,
            dest_root,
            dest_root_to_images,
            n_samples_per_class,
            copy_images=False,
            val_set_counts=10,
            create_test_set = False
    ):
        train_json_path = Path(train_json_path)
        dest_root = Path(dest_root)
        dest_root_to_images = Path(dest_root_to_images)
        train_images_dict = json.load(open(train_json_path))
        resulted_dic = {}
        for class_name, all_files in train_images_dict.items():
            random.shuffle(all_files)
            selected_files = all_files[: n_samples_per_class + val_set_counts]
            selected_files = [
                dest_root_to_images.joinpath(f"{it.replace('.JPEG', '')}.jpg").__str__()
                for it in selected_files
                if dest_root_to_images.joinpath(f"{it}.jpg").exists() or dest_root_to_images.joinpath(
                    f"{it.split('.')[0]}.jpg").exists()
            ]
            resulted_dic[class_name] = selected_files
        val_dic = dict.fromkeys(resulted_dic.keys())
        for k in resulted_dic.keys():
            val_dic[k] = [resulted_dic[k].pop() for i in range(val_set_counts)]
        if not create_test_set:
            write_set(
                    dest_root=dest_root,
                    which="train",
                    resulted_dic=resulted_dic,
                    copy_images=copy_images,
                    pool=DatasetBuilder._worker_pool,
            )
            write_set(
                    dest_root=dest_root,
                    which="val",
                    resulted_dic=val_dic,
                    copy_images=copy_images,
                    pool=DatasetBuilder._worker_pool,
            )
        else:
            write_set(
                    dest_root=dest_root,
                    which="test",
                    resulted_dic=resulted_dic,
                    copy_images=copy_images,
                    pool=DatasetBuilder._worker_pool,
            )

        logger.info(f"images are under {dest_root}")


if __name__ == "__main__":
    fire.Fire(DatasetBuilder)
