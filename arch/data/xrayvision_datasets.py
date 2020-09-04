import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from torchvision.datasets.folder import default_loader

from . import bbox_utils
from ..utils import DotDict

thispath = os.path.dirname(os.path.realpath(__file__))


class DatasetBase(torch.utils.data.Dataset):
    @classmethod
    def get_train_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 14

        train_tx = tv.transforms.Compose([
            # By default the imageFolder loads images with 3 channels
            # and we expect the image to be grayscale.
            # So let's transform the image to grayscale
            tv.transforms.Grayscale(),
            tv.transforms.Resize(orig_size),
            tv.transforms.RandomCrop((orig_size, orig_size)),
            # tv.transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ])
        return train_tx

    @classmethod
    def get_val_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 56
        val_tx = tv.transforms.Compose([
            tv.transforms.Grayscale(),
            tv.transforms.Resize(orig_size),
            tv.transforms.CenterCrop((orig_size, orig_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ])
        return val_tx

    def make_loader(self, batch_size, shuffle, workers):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True, drop_last=False)


class Kaggle_Dataset(DatasetBase):
    """
    Modify from torchxrayvision github repo:
    https://github.com/mlmed/torchxrayvision

    RSNA Pneumonia Detection Challenge

    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """
    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(thispath, "kaggle_stage_2_train_labels.csv.zip"),
                 dicomcsvpath=os.path.join(thispath, "kaggle_stage_2_train_images_dicom_headers.csv.gz"),
                 views=["PA", "AP"],
                 transform=None,
                 include='all',
                 **kwargs):
        assert include in ['all', 'pos', 'neg', 'pos_bbox', 'all_bbox']

        super().__init__()
        self.imgpath = imgpath
        self.transform = transform
        # self.pathologies = ["Pneumonia", "Lung Opacity"]
        # self.pathologies = sorted(self.pathologies)

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        self.dicomcsvpath = dicomcsvpath
        self.dicomcsv = pd.read_csv(self.dicomcsvpath, index_col="PatientID")
        self.csv = self.csv.join(self.dicomcsv, on="patientId")

        self.MAXVAL = 255  # Range [0 255]
        if type(views) is not list:
            views = [views]
        self.views = views
        # Remove images with view position other than specified
        self.csv = self.csv[self.csv['ViewPosition'].isin(self.views)]

        self.include = include
        if include in ['pos', 'pos_bbox']:
            self.csv = self.csv[self.csv['Target'] == 1]
        if include == 'neg':
            self.csv = self.csv[self.csv['Target'] == 0]

        # Get our classes.
        self.labels = self.csv["Target"].values

        if include in ['pos_bbox', 'all_bbox']:
            self.xs = torch.from_numpy(np.nan_to_num(
                self.csv['x'].values, copy=False, nan=-1.).astype(int))
            self.ys = torch.from_numpy(np.nan_to_num(
                self.csv['y'].values, copy=False, nan=-1.).astype(int))
            self.ws = torch.from_numpy(np.nan_to_num(
                self.csv['width'].values, copy=False, nan=-1.).astype(int))
            self.hs = torch.from_numpy(np.nan_to_num(
                self.csv['height'].values, copy=False, nan=-1.).astype(int))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        imgid = self.csv['patientId'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + '.jpg')

        img = default_loader(img_path)
        target = self.labels[idx]

        if self.include not in ['pos_bbox', 'all_bbox']:
            if self.transform is not None:
                img = self.transform(img)
            return img, target

        new_result = DotDict()
        new_result['imgs'] = img
        new_result['xs'] = self.xs[idx]
        new_result['ys'] = self.ys[idx]
        new_result['ws'] = self.ws[idx]
        new_result['hs'] = self.hs[idx]

        if self.transform is not None:
            new_result = self.transform(new_result)

        return new_result, target

    @property
    def is_bbox_folder(self):
        return self.include in ['pos_bbox', 'all_bbox']

    @classmethod
    def get_train_bbox_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 56

        train_tx = tv.transforms.Compose([
            bbox_utils.Grayscale(),
            bbox_utils.Resize(orig_size),
            bbox_utils.RandomCrop((orig_size, orig_size)),
            # data_utils.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5,), (0.5,)),
        ])
        return train_tx

    @classmethod
    def get_val_bbox_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 56
        val_tx = tv.transforms.Compose([
            bbox_utils.Grayscale(),
            bbox_utils.Resize(orig_size),
            bbox_utils.CenterCrop((orig_size, orig_size)),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5,), (0.5,)),
        ])
        return val_tx

    def make_loader(self, batch_size, shuffle, workers):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True, drop_last=False,
            collate_fn=bbox_utils.bbox_collate)


class CheX_Dataset(DatasetBase):
    """
    Just use the pneumonia label. Total 17815 patients with 0.159 positive rate.

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """
    def __init__(self,
                 imgpath='../datasets/CheXpert/',
                 csvpath='../datasets/CheXpert/map.csv',
                 views=["PA", "AP"],
                 transform=None,
                 unique_patients=True):
        # self.pathologies = ["Enlarged Cardiomediastinum",
        #                     "Cardiomegaly",
        #                     "Lung Opacity",
        #                     "Lung Lesion",
        #                     "Edema",
        #                     "Consolidation",
        #                     "Pneumonia",
        #                     "Atelectasis",
        #                     "Pneumothorax",
        #                     "Pleural Effusion",
        #                     "Pleural Other",
        #                     "Fracture",
        #                     "Support Devices"]
        # self.pathologies = sorted(self.pathologies)

        super().__init__()
        self.imgpath = imgpath
        self.transform = transform

        # Load data
        self.csvpath = csvpath
        df = pd.read_csv(self.csvpath)

        if type(views) is not list:
            views = [views]
        self.views = views

        # Remove images with view position other than specified
        idx_pa = df["AP/PA"].isin(self.views)
        df = df[idx_pa]

        # Unique patients
        if unique_patients:
            df["PatientID"] = df["Path"].str.extract(pat='(patient\d+)')
            df = df.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = (df["No Finding"] == 1)
        df.Pneumonia.loc[healthy] = 0
        df = df[(~df.Pneumonia.isna()) & (df.Pneumonia != -1)]

        self.labels = df.Pneumonia.values.astype(int)
        self.paths = df.Path.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = pjoin(self.imgpath, self.paths[idx])

        img = default_loader(img_path)
        target = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @property
    def is_bbox_folder(self):
        return False

    def make_loader(self, batch_size, shuffle, workers):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True, drop_last=False)


class MIMIC_Dataset(DatasetBase):
    """
    Only extract pneumonia labels. 51105 patients with 6.65% positive rate.

    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S. MIMIC-CXR: A large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self,
                 mimic_folder='../datasets/MIMIC-CXR/',
                 transform=None,
                 unique_patients=True):

        super().__init__()
        self.mimic_folder = mimic_folder
        self.transform = transform

        # Load data
        def load_data(mode='train'):
            csvpath = pjoin(mimic_folder, f'{mode}.csv')
            df = pd.read_csv(csvpath)
            df['subject_id'] = df["path"].apply(lambda x: int(x.split('/')[1][1:]))
            if unique_patients:
                df = df.groupby("subject_id").first().reset_index()

            # Get our classes.
            healthy = (df["No Finding"] == 1)
            df.Pneumonia.loc[healthy] = 0
            df = df[(~df.Pneumonia.isna()) & (df.Pneumonia != -1)]

            return df.path.values, df.Pneumonia.values.astype(int)

        train_paths, train_labels = load_data('train')
        val_paths, val_labels = load_data('valid')

        self.paths = np.concatenate([train_paths, val_paths], axis=0)
        self.labels = np.concatenate([train_labels, val_labels], axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = pjoin(self.mimic_folder, self.paths[idx])
        img = default_loader(img_path)
        target = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, target


class NIH_Dataset(DatasetBase):
    """
    Pneumonia with 30805 patients and 3.3% positive rate

    NIH ChestX-ray8 dataset

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 nih_folder='../datasets/NIH/',
                 transform=None,
                 unique_patients=True):

        super().__init__()
        self.nih_folder = nih_folder
        self.transform = transform

        # Load data
        csvpath = pjoin(nih_folder, 'preprocessed.csv')
        df = pd.read_csv(csvpath)
        if unique_patients:
            # Take the image with penumonia as 1 to increase positive cases
            df = df.groupby("Patient ID").apply(
                lambda x: x[x.Pneumonia == 1].iloc[0]
                if (x.Pneumonia == 1).any()
                else x.iloc[0])

        self.paths = df['Image Index'].values
        self.labels = df.Pneumonia.values.astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = pjoin(self.nih_folder, 'images', self.paths[idx])
        img = default_loader(img_path)
        target = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
