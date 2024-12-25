import gzip
import json

from decision.glip_utils import get_projection


if __name__ == "__main__":
    # gz_file_path: str = "habitat-challenge-data/data/val/val.json.gz"
    gz_file_path: str = "habitat-challenge-data/data/hm3d/v2/val_mini.json.gz"
    projection: dict = get_projection(gz_file_path)
    print(projection)
    assert isinstance(projection, dict), "Error happened in reading the gz file"