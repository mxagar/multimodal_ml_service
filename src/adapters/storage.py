"""Module for interacting with cloud storage services.

Specifically, a class `ObjectStorage` is defined,
which represents an S3 bucket.

Boto3 is used to interact with AWS S3.

NOTE: The class `ObjectStorage` wraps stateless functions
defined in this module to provide both an object-oriented interface and a functional interface.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import os
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm
import pathlib

import boto3
import botocore
import pandas as pd

from src.core import SRC_PATH

def create_s3_client() -> boto3.client:
    """Create an S3 client."""
    # Get credentials from .env file
    load_dotenv(override=True, dotenv_path=str(SRC_PATH / ".env"))
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", default=None)
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", default=None)
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", default=None)
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", default=None)
    if AWS_ACCESS_KEY_ID is None \
        or AWS_SECRET_ACCESS_KEY is None \
        or AWS_SESSION_TOKEN is None \
        or AWS_DEFAULT_REGION is None:
        error_message = "Missing AWS credentials in .env file"
        raise ValueError(error_message)

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_DEFAULT_REGION
    )

    return s3_client


def print_buckets(s3_client: Optional[boto3.client] = None) -> None:
    """List all available S3 buckets."""
    if s3_client is None:
        s3_client = create_s3_client()
    buckets = s3_client.list_buckets()
    print("Available Buckets:")  # noqa: T201
    for bucket in buckets["Buckets"]:
        print(f"  - {bucket['Name']}")  # noqa: T201


def create_s3_bucket(bucket_name: str, s3_client: Optional[boto3.client] = None) -> None:
    """Create an S3 bucket."""
    if s3_client is None:
        s3_client = create_s3_client()
    region = s3_client.meta.region_name

    try:
        _ = s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={
                "LocationConstraint": region
            }
        )
        print(f"Bucket {bucket_name} created successfully in region {region}.")  # noqa: T201
    except botocore.exceptions.ClientError as e:
        print(f"Error: {e.response['Error']['Message']}")  # noqa: T201


def delete_s3_bucket(bucket_name: str, s3_client: Optional[boto3.client] = None) -> None:
    """Delete an S3 bucket."""
    if s3_client is None:
        s3_client = create_s3_client()

    try:
        _ = s3_client.delete_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} deleted successfully.")  # noqa: T201
    except botocore.exceptions.ClientError as e:
        print(f"Error: {e.response['Error']['Message']}")  # noqa: T201


def get_file_entries_in_bucket(bucket_name: str,
                               s3_client: Optional[boto3.client] = None,
                               debug: bool = False) -> list:
    """Get a list of file entries in an S3 bucket."""
    if s3_client is None:
        s3_client = create_s3_client()
    file_entries = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in tqdm(paginator.paginate(Bucket=bucket_name),
                         desc="Retrieving files from bucket",
                         unit="page",
                         disable=not debug):
            if "Contents" in page:
                file_entries.extend(page["Contents"])
    except Exception as e:
        print(f"Error: {e}")  # noqa: T201

    return file_entries


def print_filenames_in_bucket(bucket_name: str,
                              s3_client: Optional[boto3.client] = None) -> None:
    """Print filenames in an S3 bucket."""
    file_entries = get_file_entries_in_bucket(bucket_name, s3_client=s3_client)
    if len(file_entries) > 0:
        for file in file_entries:
            print(file["Key"])  # noqa: T201


def organize_file_entries_into_dataframe(file_entries: list) -> pd.DataFrame:
    """Create a DataFrame from a list of file entries."""
    data = []
    for entry in file_entries:
        filepath = entry["Key"]
        filepath_elements =  filepath.split("/")
        token = filepath_elements[-2]
        filename = filepath_elements[-1]
        modified = entry["LastModified"]
        size = entry["Size"]
        data.append({"token": token,
                     "filename": filename,
                     "filepath": filepath,
                     "last_modified": modified,
                     "size": size})

    df = pd.DataFrame(data)

    return df


def get_bucket_files_dataframe(bucket_name: str,
                               s3_client: Optional[boto3.client] = None) -> pd.DataFrame:
    """Get a DataFrame with the files in an S3 bucket."""
    file_entries = get_file_entries_in_bucket(bucket_name, s3_client=s3_client)
    df = organize_file_entries_into_dataframe(file_entries=file_entries)

    return df


def download_file_from_s3(bucket_name: str,
                          object_name: str,
                          file_name: Optional[str] = None,
                          s3_client: Optional[boto3.client] = None) -> None:
    """Download a file from an S3 bucket."""
    if s3_client is None:
        s3_client = create_s3_client()
    if file_name is None:
        file_name = object_name

    # Ensure all directories in the file path are created
    pathlib.Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    try:
        s3_client.download_file(bucket_name, object_name, file_name)
    except Exception as e:
        print(f"Error downloading file: {e}")  # noqa: T201


def upload_file_to_s3(bucket_name: str,
                      file_name: str,
                      object_name: Optional[str] = None,
                      s3_client: Optional[boto3.client] = None) -> None:
    """Upload a file to an S3 bucket"""
    if s3_client is None:
        s3_client = create_s3_client()
    if object_name is None:
        object_name = file_name

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
    except Exception as e:
        print(f"Error uploading file: {e}")  # noqa: T201


def delete_file_from_s3(bucket_name: str,
                        object_name: str,
                        s3_client: Optional[boto3.client] = None) -> None:
    """Delete a file from an S3 bucket."""
    if s3_client is None:
        s3_client = create_s3_client()

    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_name)
    except Exception as e:
        print(f"Error deleting file: {e}")  # noqa: T201


class ObjectStorage:
    """A class representing a cloud storage repository.

    Currently, it implements the connection to an S3 bucket.
    It performs operations like:
    - retrieving a DataFrame of file entries,
    - uploading files,
    - downloading files,
    - and deleting files.

    This class uses a class-level cache to store the S3 client, ensuring that any
    class instance has access to the same client.
    """
    _cached_client = None  # Class-level cache, similar to a Singleton

    def __init__(self, name: str):
        """Initialize the ObjectStorage with the given bucket name.

        Generates the S3 client and assigns it as an attribute.

        Args:
            name (str): Name of the S3 bucket, e.g. 's3://my-bucket'.
        """
        self.name = name.split("://")[-1]  # Remove the protocol prefix if present, i.e., s3://
        if ObjectStorage._cached_client is None:
            ObjectStorage._cached_client = create_s3_client()
        self.client = ObjectStorage._cached_client

    def get_dataframe(self) -> pd.DataFrame:
        """Retrieve a DataFrame containing metadata of all files in the cloud repository (S3 bucket).

        Returns:
            pd.DataFrame: DataFrame containing file metadata.
        """
        return get_bucket_files_dataframe(bucket_name=self.name, s3_client=self.client)

    def download_file(self, object_name: str, file_name: Optional[str|pathlib.Path] = None) -> None:
        """Download a file from the cloud repository (S3 bucket).

        Args:
            object_name (str): The key of the file in the bucket.
            file_name (Optional[str|pathlib.Path]): Local file path to save the downloaded file.
        """
        if isinstance(file_name, pathlib.Path):
            file_name = str(file_name)

        download_file_from_s3(
            bucket_name=self.name,
            object_name=object_name,
            file_name=file_name,
            s3_client=self.client
        )

    def upload_file(self, file_name: str|pathlib.Path, object_name: Optional[str] = None) -> None:
        """Upload a file to the cloud repository (S3 bucket).

        Args:
            file_name (str|pathlib.Path): Local file path of the file to upload.
            object_name (Optional[str]): Key to assign to the file in the bucket.
        """
        if isinstance(file_name, pathlib.Path):
            file_name = str(file_name)

        upload_file_to_s3(
            bucket_name=self.name,
            file_name=file_name,
            object_name=object_name,
            s3_client=self.client
        )

    def delete_file(self, object_name: str) -> None:
        """Delete a file from the cloud repository (S3 bucket).

        Args:
            object_name (str): The key of the file to delete in the bucket.
        """
        delete_file_from_s3(
            bucket_name=self.name,
            object_name=object_name,
            s3_client=self.client
        )
