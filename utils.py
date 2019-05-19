import glob
from pathlib import Path


def counts_file(root_dir):
    root_dir = Path(root_dir)
    # return len(list(glob.iglob(root_dir.joinpath('*/*').__str__(), recursive=True)))
    return sum(1 for i in glob.iglob(root_dir.joinpath('*/*').__str__()))


if __name__ == '__main__':
    d = r'/l/workspace/dataset/deeppg/train/images'
    print(counts_file(d))