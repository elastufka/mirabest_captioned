# %pip install  -U -q git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/trl.git datasets peft accelerate


import os
import sys
import numpy as np
import torch.utils.data as data
from datasets import Dataset, Features, Value, Image
from functools import partial
import time
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image as PILImage
from torchvision.datasets.utils import download_url, check_integrity
from google import genai
from environment import GEMINI_FLASH_2_API_KEY
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
import PIL
from google.generativeai import GenerativeModel, configure
from google.genai import types


class MiraBest_full(data.Dataset):
    """​
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-full.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    base_folder = "batches"
    url = "http://www.jb.man.ac.uk/research/MiraBest/full_dataset/MiraBest_full_batches.tar.gz"
    filename = "MiraBest_full_batches.tar.gz"
    tgz_md5 = "965b5daa83b9d8622bb407d718eecb51"
    train_list = [
        ["data_batch_1", "b15ae155301f316fc0b51af16b3c540d"],
        ["data_batch_2", "0bf52cc1b47da591ed64127bab6df49e"],
        ["data_batch_3", "98908045de6695c7b586d0bd90d78893"],
        ["data_batch_4", "ec9b9b77dc019710faf1ad23f1a58a60"],
        ["data_batch_5", "5190632a50830e5ec30de2973cc6b2e1"],
        ["data_batch_6", "b7113d89ddd33dd179bf64cb578be78e"],
        ["data_batch_7", "626c866b7610bfd08ac94ca3a17d02a1"],
    ]

    test_list = [
        ["test_batch", "5e443302dbdf3c2003d68ff9ac95f08c"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "e1b5450577209e583bc43fbf8e851965",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = PILImage.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

class MBFRConfident(MiraBest_full):
    """
    Child class to load only confident FRI (0) & FRII (1)
    [100, 102, 104] and [200, 201]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRConfident, self).__init__(*args, **kwargs)

        fr1_list = [0, 1, 2]
        fr2_list = [5, 6]
        exclude_list = [3, 4, 7, 8, 9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class~0
            targets[fr2_mask] = 1  # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0  # set all FRI to Class 0
            targets[fr2_mask] = 1  # set all FRII to Class 1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()

# (1) generator that yields your 4-field dicts
def gen(ds):
    for img, orig_lbl in ds:
        # map numeric label → "FR-I" / "FR-II" / ""
        if orig_lbl in fr1_list:
            lbl = "FR-I"
        elif orig_lbl in fr2_list:
            lbl = "FR-II"
        else:
            continue
        yield {
            "image":            img,
            "query":            "",
            "label":            lbl,
            "human_or_machine": ""
        }

# Convert to JPEG bytes
def PIL_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return image_bytes

def zero_shot_predict(
        test_dataset,
        query_idx,
        device="cuda",
        print_text=False
    ):
    imgq = test_dataset[query_idx]["image"]
    imgq_bytes = PIL_to_bytes(imgq)

    contents = [
        system_message,
        types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"),
        query_text,
    ]

    answer = client.models.generate_content(
        model=model_name,
        contents=contents,
    )
    if print_text:
        print(answer.text)
    answer = answer.text
    return answer

if __name__ == "__main__":
    fr1_list = [0, 1, 2]
    fr2_list = [5, 6]

    # (2) declare the exact schema you want
    features = Features({
        "image":            Image(decode=True),
        "query":            Value("string"),
        "label":            Value("string"),
        "human_or_machine": Value("string"),
    })

    # (3) build the Dataset
    train_dataset = Dataset.from_generator(partial(gen, train_ds), features=features)
    test_dataset = Dataset.from_generator(partial(gen, test_ds), features=features)

    # (4) check
    print(train_dataset)
    #print(test_dataset)

    client = genai.Client(api_key=GEMINI_FLASH_2_API_KEY)
    model_name="gemini-2.5-flash-preview-05-20"

    # 1) Your system & query
    #system_message = """You are acting as an expert observational radio astronomer. Examine the following image and describe it in a concise manner. Look for bright centroids of flux and where in the image (left, right, top, bottom, center) they are located. Are there any lobes or jets associated with the flux centroids? From where do they originate and where do they end? Are there any sources in the image that are not associated with the central galaxy? Only describe morphology, leaving out any assumptions about the nature of the emission. Use 50 words or less.
    #"""
    #query_text     = "Please use the above instructions to describe this image."
    system_message = """"""
    query_text     = "Please use 50 words or less to write a caption describing this image."

    captions = []
    index= []
    dataset = train_dataset #or test
    idx = 0 #if needed to restart due to API limits
    for i in tqdm(range(len(dataset)), desc="Generating captions"):
        if i >= idx:
        print(i)
        try:
            pred = zero_shot_predict(
                dataset,
                query_idx=i,
                device="cuda")
            captions.append(pred)
            index.append(i)
        except Exception as e:
            if "429" in e: #Out of resources error
            print(e)
            break
            #df = pd.DataFrame({"index":range(idx, idx+len(captions)),"caption":captions})
            #df.to_csv(f"train_captions{idx}-{idx+len(captions)}.csv")
        time.sleep(5)
        # strip off any leading "Answer:" if you included that
        if pred.lower().startswith("answer"):
            pred = pred.split(":",1)[1].strip()
    
    df = pd.DataFrame({"index":index,"caption":captions})
    df.to_csv(f"test_captions_blind.csv")