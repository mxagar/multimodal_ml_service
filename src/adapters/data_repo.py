"""A repository class to interface with a remote and distributed data storage system.

This module contains the DataRepository class, which uses both ObjectStorage and AnnotationProject.

DataRepository should be used by Dataset, which creates a local dataset from a remote one.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pathlib
import ast
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.config.config import AnnotationProjectConfig
from src.adapters.storage import ObjectStorage
from src.adapters.annotations import AnnotationProject
from src.core import DATA_PATH


EXCLUDE_SUBSTRINGS = ["_trans", ".csv"]


class DataRepository:
    """A repository class to interface with a remote and distributed data storage system.

    It makes possible to reproduce locally a dataset stored in a remote storage system.
    DataRepository is used by Dataset, which creates a local dataset from a remote one.

    DataRepository contains:
    - an ObjectStorage instance to interact with the S3 bucket.
    - and an AnnotationProject instance to interact with the LabelStudio project.

    Responsibilities:
    - Fetch and create a dataframe of the contents in an ObjectStorage (S3 bucket).
    - Fetch and create a dataframe of the contents in an AnnotationProject (LS).
    - Download files from ObjectStorage (S3), either individually or in bulk, to a local directory.
    """

    def __init__(self, config: AnnotationProjectConfig,
                 local_data_path: Optional[str|pathlib.Path] = None) -> None:
        """Initialize the DataRepository.

        Args:
            config (AnnotationProjectConfig): The configuration for the AnnotationProject.
            local_data_path (str|pathlib.Path): The local data path to store the downloaded files.
        """
        self.config = config
        self.storage_bucket = ObjectStorage(self.config.sample_storage_uri)
        self.annotation_project = AnnotationProject(self.config)
        self.local_data_path = self._get_local_data_path(local_data_path)
        self.samples_storage_csv_filename = self.local_data_path / "samples_storage.csv"
        self.samples_annotated_csv_filename = self.local_data_path / "samples_annotated.csv"

        # These attributes are set after calling the download_data method.
        # These dataframes are cached as attributes.
        self.storage_dataframe: pd.DataFrame = None
        self.annotation_dataframe: pd.DataFrame = None

    def _get_local_data_path(self, path: Optional[str|pathlib.Path] = None) -> pathlib.Path:
        """Set the local data path for the repository.

        Returns:
            pathlib.Path: The local data path.
        """
        # Set initial local data path
        local_data_path = DATA_PATH
        if path is not None:
            local_data_path = pathlib.Path(path)
        # Extend with URI
        local_data_path /= pathlib.Path(str(self.config.sample_storage_uri).replace("://", "/"))
        # Create directory
        local_data_path.mkdir(parents=True, exist_ok=True)

        return local_data_path

    def get_storage_dataframe(self,
                              output_path: Optional[str|pathlib.Path] = None,
                              persist: bool = True,
                              retry: bool = False) -> pd.DataFrame:
        """Get a dataframe of the contents in the ObjectStorage (S3) and persist it.

        Args:
            output_path (str|pathlib.Path): The path to save the CSV file.
                If None, it saves to samples_storage.csv.
            persist (bool): Whether to save the DataFrame to a CSV file. Defaults to True.
            retry (bool): Whether to retry the DataFrame. Defaults to False.
                The DataFrame is cached in the storage_dataframe attribute.

        Returns:
            pd.DataFrame: The DataFrame of S3 contents.
        """
        if self.storage_dataframe is None or retry:
            if output_path is None:
                output_path = self.samples_storage_csv_filename
            df = None
            if output_path.exists() and not retry:
                df = pd.read_csv(str(output_path))
            else:
                df = self.storage_bucket.get_dataframe()
            if persist:
                df.to_csv(str(output_path), index=False)
            self.storage_dataframe = df

        return self.storage_dataframe

    def get_annotation_dataframe(self,
                                 output_path: Optional[str|pathlib.Path] = None,
                                 only_annotated: bool = True,
                                 persist: bool = True,
                                 retry: bool = False) -> pd.DataFrame:
        """Create a dataframe with the sample filepaths and labels from the AnnotationProject (LS).

        Args:
            output_path (str|pathlib.Path): The path to save the CSV file.
                If None, it saves to samples_annotated.csv.
            only_annotated (bool): Whether to include only annotated tasks. Defaults to True.
            persist (bool): Whether to save the DataFrame to a CSV file. Defaults to True.
            retry (bool): Whether to retry the DataFrame. Defaults to False.
                The DataFrame is cached in the annotation_dataframe attribute.

        Returns:
            pd.DataFrame: The DataFrame of LabelStudio labels.
        """
        if self.annotation_dataframe is None or retry:
            if output_path is None:
                output_path = self.samples_annotated_csv_filename
            df = None
            if output_path.exists() and not retry:
                df = pd.read_csv(str(output_path))
                df["labels"] = df["labels"].apply(ast.literal_eval)
            else:
                df = self.annotation_project.get_annotation_dataframe(only_annotated=only_annotated)
            if persist:
                df.to_csv(str(output_path), index=False)
            self.annotation_dataframe = df

        return self.annotation_dataframe

    def download_single_sample(self,
                               object_name: str,
                               local_object_path: Optional[str|pathlib.Path] = None) -> None:
        """Download a single file/sample from the ObjectStorage if it doesn't exist locally.

        Args:
            object_name (str): The object name in ObjectStorage (S3 object key).
            local_object_path (str|pathlib.Path): The local path to save the downloaded file.
                If None, it saves to the local_data_path.
        """
        if local_object_path is None:
            local_object_path = self.local_data_path / object_name
        if not local_object_path.exists():
            self.storage_bucket.download_file(object_name, str(local_object_path))

    def download_multiple_samples(self,
                                  object_names: List[str],
                                  local_directory: Optional[str|pathlib.Path] = None,
                                  debug: bool = False) -> None:
        """Download multiple files/samples from the ObjectStorage if it doesn't exist locally.

        Args:
            object_names (List[str]): List of object names in ObjectStorage (S3 object keys).
            local_directory (str|pathlib.Path): The local directory to save the downloaded files.
            debug (bool): First filter missing files and show download progress. Defaults to False.
        """
        if local_directory is None:
            local_directory = self.local_data_path
        local_directory.mkdir(parents=True, exist_ok=True)

        if debug:
            # First check the missing amount
            objects_to_download = []
            for object_name in object_names:
                local_object_path = local_directory / object_name
                if not local_object_path.exists():
                    objects_to_download.append(object_name)
            print(f"Dataset: {self.config.sample_storage_uri}")  # noqa: T201
            print(f"Samples to be downloaded: {len(objects_to_download)}")  # noqa: T201
            # Then load the filtered list
            for object_name in tqdm(objects_to_download, desc="Downloading samples"):
                local_object_path = local_directory / object_name
                self.download_single_sample(object_name, local_object_path)
        else:
            # Check of existence is handled by download_file()
            for object_name in object_names:
                local_object_path = local_directory / object_name
                self.download_single_sample(object_name, local_object_path)

    def download_data(self,
                      download_samples: Optional[bool] = None,
                      retry: bool = False,
                      debug: bool = False) -> None:
        """Download the data from the remote storage system: dataframes and samples.

        Args:
            download_samples (bool): Whether to download the files/samples. Defaults to None.
                If None, it uses the value in the AnnotationProjectConfig.
            retry (bool): Whether to retry the dataframes. Defaults to False.
                If True, dataframes are downloaded again, even if they are cached in memory.
            debug (bool): Whether to show debug information. Defaults to False.
        """
        if download_samples is None:
            download_samples = self.config.download_samples

        df_storage = self.get_storage_dataframe(retry=retry)
        df_annotations = self.get_annotation_dataframe(retry=retry)

        if download_samples:
            storage_objects = df_storage["filepath"]
            annotated_objects = df_annotations[df_annotations["filepath"].isin(storage_objects)]["filepath"]
            self.download_multiple_samples(annotated_objects.tolist(), self.local_data_path, debug=debug)

    def get_local_data_filepaths(
        self,
        exclude_substrings: Optional[List[str]] = EXCLUDE_SUBSTRINGS) -> List[str]:
        """Get the list of local filepaths in the local data path.

        Args:
            exclude_substrings (List[str]): Files which contain these substrings
                will be excluded from the results. Defaults to EXCLUDE_SUBSTRINGS,
                i.e.: .csv, _trans, etc.

        Returns:
            List[str]: The list of all local filepaths in local_data_path.
        """
        if isinstance(self.local_data_path, str):
            self.local_data_path = pathlib.Path(self.local_data_path)
        filepaths = [
            str(path) for path in self.local_data_path.rglob("*")
            if path.is_file() and not any(substring in path.name for substring in exclude_substrings)
        ]
        return filepaths

    def get_local_data_filepaths_and_labels(
        self,
        only_labels: Optional[List[str]] = None,
        exclude_substrings: Optional[List[str]] = EXCLUDE_SUBSTRINGS,
    ) -> Tuple[List[str], List[str]]:
        """Get the list of local filepaths and labels in the local data path.

        Note: It is possible that one sample has more than one label; this is resolved
        by returning the same filepath multiple times with different labels.

        Args:
            only_labels (List[str]): The list of labels to filter the results. Defaults to None.
                If not None, only the samples with the labels in only_labels are picked.
            exclude_substrings (List[str]): Files which contain these substrings
                will be excluded from the results. Defaults to EXCLUDE_SUBSTRINGS,
                i.e.: .csv, _trans, etc.

        Returns:
            Tuple[List[str], List[str]]: The list of all local (absolute) filepaths
                and their labels in local_data_path.
        """
        def _find_matching_indices(local_filepaths: List[pathlib.Path],
                                   df_annotations: pd.DataFrame,
                                   filepath_column: str = "filepath") -> List[Optional[int]]:
            local_filepaths_str = [str(path).replace("\\", "/") for path in local_filepaths]
            indices = []
            for local_filepath in local_filepaths_str:
                matches = df_annotations[filepath_column].apply(lambda x: x in local_filepath)  # noqa: B023
                matching_index = matches.idxmax() if matches.any() else None
                indices.append(matching_index)
            return indices

        # Get local filepaths and annotations
        df_annotations = self.get_annotation_dataframe()
        local_filepaths = self.get_local_data_filepaths(exclude_substrings=exclude_substrings)
        # Match local file paths and annotations
        indices = _find_matching_indices(local_filepaths=local_filepaths,
                                         df_annotations=df_annotations,
                                         filepath_column="filepath")
        # Filter out the None indices and get labels of matched filepaths
        filepaths = []
        labels = []
        for i, filepath in enumerate(local_filepaths):
            if indices[i] is not None:
                sample_labels = df_annotations.loc[i, "labels"]
                for label in sample_labels:
                    if only_labels is not None and label not in only_labels:
                        continue
                    filepaths.append(filepath)
                    labels.append(label)

        return filepaths, labels

    def reorganize_local_files(self) -> None:
        """Reorganize the local files into a directory structure based on the labels."""
        # TODO: Implement this method
        _ = self.get_annotation_dataframe()
