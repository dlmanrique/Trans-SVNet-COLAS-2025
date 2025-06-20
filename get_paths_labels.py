import os
import json
import pickle as pkl
from typing import *
from tqdm import tqdm


def open_annotations(file_path: str) -> Dict[str, Union[Dict[str, str], List[Dict[str, Union[int, str]]]]]:
    with open(os.path.join(file_path), "r") as file:
        anns = json.load(file)
    return anns


def get_img_path(dataset_path: str, images_info: List[Dict[str, Union[int, str]]]) -> List[str]:
    return [os.path.join(dataset_path, img_info["file_name"]) for img_info in images_info]


def get_labels(annotations: List[Dict[str, Union[int, str]]]) -> List[List[int]]:
    return [annotation["phases"] for annotation in annotations]


def number_frames_each(images_info: List[Dict[str, Union[int, str]]]) -> List[int]:
    num_each = []
    first = True

    for img_info in tqdm(images_info):
        if first:
            video_name = img_info["video_name"]
            first = False
            count = 1
        else:
            if video_name == img_info["video_name"]:
                count += 1
            else:
                num_each.append(count)
                video_name = img_info["video_name"]
                count = 1

    num_each.append(count)

    return num_each


def process_fold(dataset_path: str, fold_path: str, dataset_name: str) -> None:
    #Open annotations for phases (long-term)
    train_anns = open_annotations(os.path.join(fold_path, f"long_term_{dataset_name}_train.json"))
    test_anns = open_annotations(os.path.join(fold_path, f"long_term_{dataset_name}_test.json")) #-> just for Autolaparo set the valid set as test and then make inference in the real test file

    train_img_path = get_img_path(dataset_path, train_anns["images"])
    test_img_path = get_img_path(dataset_path, test_anns["images"])


    train_labels = get_labels(train_anns["annotations"])
    test_labels = get_labels(test_anns["annotations"])

    train_num_each = number_frames_each(train_anns["images"])
    test_num_each = number_frames_each(test_anns["images"])

    all_info = [
        train_img_path,
        test_img_path,
        train_labels,
        test_labels,
        train_num_each,
        test_num_each,
    ]

    os.makedirs('pkl_datasets_files', exist_ok=True)

    with open(f"pkl_datasets_files/train_test_paths_labels_{dataset_name}.pkl", "wb") as file:
        pkl.dump(all_info, file)


def main() -> None:
    base_json_path = "DATASETS/PHASES/annotations/Original_Datasets_Splits_Annotations/json_files"
    dataset_path = ""
    dataset_name = 'Autolaparo'

    process_fold(dataset_path, base_json_path, dataset_name)


if __name__ == "__main__":
    main()