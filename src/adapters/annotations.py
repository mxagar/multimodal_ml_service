"""Module to interact with a data annotation service.

Specifically, this module provides functions to interact with Label Studio.

The class `AnnotationProject` is represents a Label Studio project.

WARNING: This module depends on the API of Label Studio, which is subject to change.

NOTE: The class `AnnotationProject` wraps stateless functions
defined in this module to provide both an object-oriented interface and a functional interface.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import requests
import urllib3
from typing import Dict, Optional, List, Tuple

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import xml.etree.ElementTree as ET

from src.config.config import AnnotationProjectConfig


# Disable warning due to self-signed certificates, which your Python does not trust by default
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def run_query(url: str,
              config: Dict|AnnotationProjectConfig,
              method: str = "GET",
              payload: Optional[Dict] = None) -> Dict:
    """Run a REST API query to Label Studio and return the JSON response.

    Args:
        url (str): The API endpoint to query.
        config (Dict|AnnotationProjectConfig): The API configuration.
        method (str): The HTTP method to use (GET or POST).
        payload (Dict): The JSON payload to send in the request.

    Returns:
        Dict: The JSON response from the API.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()
    request_method = requests.get if method.lower() == "get" else requests.post
    response = request_method(
        f"{config['base_url']}/api/{url}",
        headers={"Authorization": f"Token {config.get('api_token', '')}"},
        verify=config.get("verify", False),
        timeout=config.get("timeout", 0),
        json=payload
    )
    if response.status_code == 200:  # noqa: PLR2004
        return response.json()
    else:
        raise Exception(f"Failed to retrieve {url}. Status code: {response.status_code}") from None


def get_projects(config: Dict|AnnotationProjectConfig,
                 display: bool = False) -> List[Dict]:
    """Get all projects from Label Studio.

    Args:
        config (Dict|AnnotationProjectConfig): Configuration for the API.
        display (bool): Whether to display the projects. Defaults to False.

    Returns:
        List[Dict]: List of projects in LabelStudio format.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()
    url = "projects"
    projects = run_query(url, config=config, method="GET")["results"]
    if display:
        for project in projects:
            print(f"Project ID: {project['id']}")  # noqa: T201
            print(f"Title: {project['title']}")  # noqa: T201
            print(f"Description: {project['description']}")  # noqa: T201
            print(f"Total tasks: {project['queue_total']}")  # noqa: T201
            print(f"Finished tasks: {project['finished_task_number']}")  # noqa: T201
            if "choice" in project["parsed_label_config"]:
                print(f"Labels: {project['parsed_label_config']['choice']['labels']}")  # noqa: T201
            print("-----")  # noqa: T201

    return projects


def get_tasks(config: Dict|AnnotationProjectConfig) -> List[Dict]:
    """Get all tasks for a LabelStudio project.

    Args:
        config (Dict|AnnotationProjectConfig): The API configuration.

    Returns:
        List[Dict]: The list of tasks in LabelStudio task format.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()
    project_id = config.get("project_id")
    url = f"projects/{project_id}/export?exportType=JSON&download_all_tasks=true"
    tasks = run_query(url, config=config, method="GET")
    return tasks


def get_project(config: Dict|AnnotationProjectConfig) -> Dict:
    """Get the dictionary of a LabelStudio project.

    Args:
        config (Dict|AnnotationProjectConfig): The API configuration.

    Returns:
        List[Dict]: The list of tasks in LabelStudio task format.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()
    project_id = config.get("project_id")
    url = f"projects/{project_id}"
    project = run_query(url, config=config, method="GET")
    return project


def get_annotation_labels(config: Dict|AnnotationProjectConfig,
                          project_data: Optional[Dict] = None) -> List[str]:
    """Get the annotation labels for a LabelStudio project.

    The labels are extracted from the label_config field,
    which contains an XML in the following format:

        <View>
        <Image name="image" value="$image"/>
        <Choices name="choice" toName="image">
            <Choice value="good"/>
            <Choice value="bad"/>
        </Choices>
        </View>

    Args:
        config (Dict|AnnotationProjectConfig): The API configuration.
        project_data (Dict): The project data dictionary.

    Returns:
        List[Dict]: The list of tasks in LabelStudio task format.
    """
    # Get project dictionary
    if project_data is None:
        project_data = get_project(config=config)
    # Parse the XML to extract label choices
    label_config = project_data.get("label_config", "")
    root = ET.fromstring(label_config)
    choices = []
    for choice in root.findall(".//Choice"):
        choices.append(choice.attrib.get("value"))

    return choices


def get_sample_identifiers(config: Dict|AnnotationProjectConfig,
                           project_data: Optional[Dict] = None) -> List[str]:
    """Get the sample identifiers from the project configuration.

    The sample identifier is the key in the task data JSON
    that identifies the sample type.

    Example with sample_identifiers=["image"]:

        <View>
            <Image name="image" value="$image"/>
            <Choices name="choice" toName="image">
                <Choice value="good"/>
                <Choice value="bad"/>
            </Choices>
        </View>
    """
    # Get project dictionary
    if project_data is None:
        project_data = get_project(config=config)

    # Parse the XML to check sample identifier
    label_config = project_data.get("label_config", "")
    root = ET.fromstring(label_config)
    sample_identifiers = []
    for element in root.iter():  # Iterate over all elements
        if "name" in element.attrib:  # Check if the 'name' attribute exists
            sample_identifiers.append(element.attrib["name"])  # ["image", "choice"]

    # Remove the identifier "choice"
    sample_identifiers = [name for name in sample_identifiers if name != "choice"]

    return sample_identifiers


def update_annotation_task(task_id: int,
                           labels: List[str],
                           config: Dict|AnnotationProjectConfig) -> None:
    """Update a task in Label Studio.

    NOTE: task_id is not the same as task_idx!

    Args:
        task_id (int): The ID of the task; this is unique independently from the project.
        labels (List[str]): The list of labels to assign to the task.
        config (Dict|AnnotationProjectConfig): The API configuration.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()

    if len(labels) > 0:
        url = f"tasks/{task_id}/annotations"
        project_id = config.get("project_id")
        payload = {
            "result": [{
                "from_name": "class",
                "to_name": config.get("sample_identifier"),
                "type": "choices",
                "value": {"choices": labels}  # this is a list
            }],
            "last_action": "prediction",
            "task": task_id,
            "project": project_id
        }
        _ = run_query(url, config=config, method="POST", payload=payload)


def get_samples_paths(tasks: List[Dict],
                      config: Dict|AnnotationProjectConfig) -> List[Tuple]:
    """Get the sample paths from the tasks.

    Args:
        tasks (List[Dict]): A list of LabelStudio tasks.
        config (Dict|AnnotationProjectConfig): The API configuration.

    Returns:
        List[Tuple]: A list of tuples containing the task ID and the sample path.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()
    sample_storage_uri = config.get("sample_storage_uri")
    if not sample_storage_uri.endswith("/"):
        sample_storage_uri += "/"
    sample_identifier = config.get("sample_identifier")
    sample_paths = []
    for task in tasks:
        sample_uri = task["data"][sample_identifier]
        task_id = task["id"]
        sample_path = sample_uri.split(sample_storage_uri)[-1]
        sample_paths.append((task_id, sample_path))
    return sample_paths


def get_samples_labels(tasks: List[Dict]) -> List[Tuple]:
    """Get the labels for each sample.

    Args:
        tasks (List[Dict]): List of LabelStudio tasks.

    Returns:
        List[Tuple]: List of tuples with the task ID and the labels.
    """
    samples_labels = []
    for task in tasks:
        task_annotations = task["annotations"]
        task_labels = []
        for annotation in task_annotations:
            if not annotation["was_cancelled"] and len(annotation["result"]) > 0:
                task_labels.extend(annotation["result"][0]["value"]["choices"])
            else:
                task_labels.extend([])
        task_labels = list(set(task_labels))
        samples_labels.append((task["id"], task_labels))
    return samples_labels


def get_annotation_project_dataframe(config: Dict|AnnotationProjectConfig,
                                     only_annotated: bool = False) -> pd.DataFrame:
    """Create a pandas DataFrame with task_id, sample_path, and labels.

    Args:
        config (Dict|AnnotationProjectConfig): The API configuration.
        only_annotated (bool): If True, only include tasks with at least one label.

    Returns:
        pd.DataFrame: DataFrame with columns 'task_id', 'sample_path', and 'labels'.
    """
    if isinstance(config, AnnotationProjectConfig):
        config = config.model_dump()
    # Get annotation tasks
    tasks = get_tasks(config=config)
    # Get sample paths and labels
    samples_paths = get_samples_paths(tasks, config=config)
    samples_labels = get_samples_labels(tasks)

    # Combine paths and labels into a DataFrame
    df_paths = pd.DataFrame(samples_paths, columns=["task_id", "filepath"])
    df_labels = pd.DataFrame(samples_labels, columns=["task_id", "labels"])

    # Merge the DataFrames on task_id
    dataset_df = pd.merge(df_paths, df_labels, on="task_id")

    # Reorganize columns
    # filepath is token/filename
    dataset_df["project_id"] = config.get("project_id")
    dataset_df["filename"] = dataset_df["filepath"].apply(lambda x: x.split("/")[-1])
    dataset_df["token"] = dataset_df["filepath"].apply(lambda x: x.split("/")[-2])
    dataset_df = dataset_df[["task_id", "project_id", "token", "filename", "filepath", "labels"]]

    # Filter for only annotated tasks if required
    if only_annotated:
        dataset_df = dataset_df[dataset_df["labels"].apply(lambda x: len(x) > 0)]

    # Reset index of the DataFrame
    dataset_df.reset_index(drop=True, inplace=True)

    # One-hot encode the labels
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(dataset_df["labels"])
    one_hot_labels_df = pd.DataFrame(one_hot_labels, columns=[f"label_{label}" for label in mlb.classes_])
    dataset_df = pd.concat([dataset_df, one_hot_labels_df], axis=1)

    return dataset_df


class AnnotationProject:
    """A class to manage a Label Studio annotation project."""

    def __init__(self, config: AnnotationProjectConfig):
        """Initialize the AnnotationProject with the given attributes.

        Args:
            config (AnnotationProjectConfig): Configuration for connecting to Label Studio.
        """
        self.config = config
        self.project_data = get_project(config=self.config)
        self.labels = get_annotation_labels(config=self.config, project_data=self.project_data)
        self.num_samples = self.project_data.get("task_number", 0)
        self.num_annotated_samples = self.project_data.get("finished_task_number", 0)
        self.project_name = self.project_data.get("title", "Unknown")
        self.sample_identifier = self.config.sample_identifier
        self._check_sample_identifier()

    def _check_sample_identifier(self) -> None:
        """Check the sample identifier from the project configuration.

        The sample identifier is the key in the task data JSON
        that identifies the sample type.
        Currently, the user defines it manually,
        but it should appear in the LS project configuration.

            <View>
                <Image name="image" value="$image"/>
                <Choices name="choice" toName="image">
                    <Choice value="good"/>
                    <Choice value="bad"/>
                </Choices>
            </View>
        """
        sample_identifiers = get_sample_identifiers(config=self.config, project_data=self.project_data)
        if self.sample_identifier not in sample_identifiers:
            raise ValueError(f"Sample identifier '{self.sample_identifier}' not found!") from None

    def get_annotation_dataframe(self, only_annotated: bool = False) -> pd.DataFrame:
        """Retrieve a DataFrame containing task IDs, sample paths, and labels.

        Args:
            only_annotated (bool): If True, only include tasks with at least one label.

        Returns:
            pd.DataFrame: DataFrame with annotation data.
        """
        return get_annotation_project_dataframe(config=self.config,
                                                only_annotated=only_annotated)

    def get_annotation_tasks(self) -> List[Dict]:
        """Retrieve all tasks for the annotation project.

        Returns:
            List[Dict]: List of Label Studio tasks.
        """
        return get_tasks(config=self.config)

    def update_annotation_task(self, task_id: int, labels: List[str]) -> None:
        """Update the annotations for a specific task.

        Args:
            task_id (int): The ID of the task to update.
            labels (List[str]): The list of labels to assign to the task.
        """
        if not set(labels).issubset(set(self.labels)):
            raise ValueError("Labels must be a subset of the annotation classes.") from None

        update_annotation_task(task_id=task_id,
                               labels=labels,
                               config=self.config)
