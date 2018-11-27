# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import tarfile

import pytest
from sagemaker.tensorflow import TensorFlow

from test.integration.utils import processor, py_version  # noqa: F401

<<<<<<< HEAD
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
TF_CHECKPOINT_FILES = ['graph.pbtxt', 'model.ckpt-0.index', 'model.ckpt-0.meta']

<<<<<<< HEAD
@pytest.fixture  # noqa: F811
def py_full_version(py_version):  # noqa: F811
=======
=======
>>>>>>> Create parameter server in different thread (#129)
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
TF_CHECKPOINT_FILES = ['graph.pbtxt', 'model.ckpt-0.index', 'model.ckpt-0.meta']


@pytest.fixture
def py_full_version(py_version):
>>>>>>> Add distributed training support (#98)
    if py_version == '2':
        return '2.7'
    else:
        return '3.6'


@pytest.mark.skip_gpu
def test_py_versions(sagemaker_local_session, docker_image, py_full_version, framework_version, tmpdir):
    output_path = 'file://{}'.format(tmpdir)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'test_py_version', 'entry.py'),
                    instance_type='local',
                    instance_count=1,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
                    framework_version=framework_version,
                    output_path=output_path,
                    training_data_path=None)

    with tarfile.open(os.path.join(str(tmpdir), 'output.tar.gz')) as tar:
        output_file = tar.getmember('py_version')
        tar.extractall(path=str(tmpdir), members=[output_file])

    with open(os.path.join(str(tmpdir), 'py_version')) as f:
        assert f.read().strip() == py_full_version


@pytest.mark.skip_gpu
<<<<<<< HEAD
def test_mnist_cpu(sagemaker_local_session, docker_image, tmpdir, framework_version):
=======
def test_mnist_cpu(sagemaker_local_session, docker_image, tmpdir):
>>>>>>> Add distributed training support (#98)
    output_path = 'file://{}'.format(tmpdir)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'mnist', 'mnist.py'),
                    instance_type='local',
                    instance_count=1,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
<<<<<<< HEAD
                    framework_version=framework_version,
=======
>>>>>>> Add distributed training support (#98)
                    output_path=output_path,
                    training_data_path='file://{}'.format(
                        os.path.join(RESOURCE_PATH, 'mnist', 'data')))
    _assert_files_exist_in_tar(output_path, ['my_model.h5'])


@pytest.mark.skip_cpu
<<<<<<< HEAD
def test_gpu(sagemaker_local_session, docker_image, framework_version):
=======
def test_gpu(sagemaker_local_session, docker_image):
>>>>>>> Add distributed training support (#98)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'gpu_device_placement.py'),
                    instance_type='local_gpu',
                    instance_count=1,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
<<<<<<< HEAD
                    framework_version=framework_version,
=======
>>>>>>> Add distributed training support (#98)
                    training_data_path='file://{}'.format(
                        os.path.join(RESOURCE_PATH, 'mnist', 'data')))


@pytest.mark.skip_gpu
<<<<<<< HEAD
def test_distributed_training_cpu_no_ps(sagemaker_local_session,
                                        docker_image,
                                        tmpdir,
                                        framework_version):
    output_path = 'file://{}'.format(tmpdir)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'mnist', 'mnist_estimator.py'),
=======
def test_distributed_training_cpu_no_ps(sagemaker_local_session, docker_image, tmpdir):
    output_path = 'file://{}'.format(tmpdir)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'mnist', 'distributed_mnist.py'),
>>>>>>> Add distributed training support (#98)
                    instance_type='local',
                    instance_count=2,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
<<<<<<< HEAD
                    framework_version=framework_version,
=======
>>>>>>> Add distributed training support (#98)
                    output_path=output_path,
                    training_data_path='file://{}'.format(
                        os.path.join(RESOURCE_PATH, 'mnist', 'data-distributed')))
    _assert_files_exist_in_tar(output_path, TF_CHECKPOINT_FILES)


@pytest.mark.skip_gpu
<<<<<<< HEAD
def test_distributed_training_cpu_ps(sagemaker_local_session,
                                     docker_image,
                                     tmpdir,
                                     framework_version):
    output_path = 'file://{}'.format(tmpdir)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'mnist', 'mnist_estimator.py'),
=======
def test_distributed_training_cpu_ps(sagemaker_local_session, docker_image, tmpdir):
    output_path = 'file://{}'.format(tmpdir)
    run_tf_training(script=os.path.join(RESOURCE_PATH, 'mnist', 'distributed_mnist.py'),
>>>>>>> Add distributed training support (#98)
                    instance_type='local',
                    instance_count=2,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
<<<<<<< HEAD
                    framework_version=framework_version,
=======
>>>>>>> Add distributed training support (#98)
                    output_path=output_path,
                    hyperparameters={'sagemaker_parameter_server_enabled': True},
                    training_data_path='file://{}'.format(
                        os.path.join(RESOURCE_PATH, 'mnist', 'data-distributed')))
    _assert_files_exist_in_tar(output_path, TF_CHECKPOINT_FILES)


<<<<<<< HEAD
<<<<<<< HEAD
def run_tf_training(script,
                    instance_type,
                    instance_count,
                    sagemaker_local_session,
                    docker_image,
                    framework_version,
                    training_data_path,
                    output_path=None,
                    hyperparameters=None):

    hyperparameters = hyperparameters or {}

    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           train_instance_count=instance_count,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_local_session,
                           image_name=docker_image,
                           model_dir='/opt/ml/model',
                           output_path=output_path,
                           hyperparameters=hyperparameters,
                           base_job_name='test-tf',
                           framework_version=framework_version,
                           py_version='py3')
=======
class ScriptModeTensorFlow(Framework):
    """This class is temporary until the final version of Script Mode is released.
    """

    __framework_name__ = "tensorflow-scriptmode-beta"

    create_model = TensorFlow.create_model

    def __init__(self, py_version='py', **kwargs):
        self.requirements_file = None
        self.py_version = py_version
        self.framework_version = 'some version'
        super(ScriptModeTensorFlow, self).__init__(**kwargs)


def run_tf_training(script, instance_type, instance_count,
                    sagemaker_local_session,
                    docker_image, training_data_path, output_path=None,
                    hyperparameters={}):
    estimator = ScriptModeTensorFlow(entry_point=script,
                                     role='SageMakerRole',
                                     train_instance_count=instance_count,
                                     train_instance_type=instance_type,
                                     sagemaker_session=sagemaker_local_session,
                                     image_name=docker_image,
                                     output_path=output_path,
                                     hyperparameters=hyperparameters,
                                     base_job_name='test-tf')
>>>>>>> Add distributed training support (#98)
=======
def run_tf_training(script,
                    instance_type,
                    instance_count,
                    sagemaker_local_session,
                    docker_image, training_data_path, output_path=None,
                    hyperparameters=None):

    hyperparameters = hyperparameters or {}

    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           train_instance_count=instance_count,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_local_session,
                           image_name=docker_image,
                           model_dir='/opt/ml/model',
                           output_path=output_path,
                           hyperparameters=hyperparameters,
                           base_job_name='test-tf',
                           framework_version='1.11.0',
                           py_version='py3')
>>>>>>> Create parameter server in different thread (#129)

    estimator.fit(training_data_path)


def _assert_files_exist_in_tar(output_path, files):
    if output_path.startswith('file://'):
        output_path = output_path[7:]
    model_file = os.path.join(output_path, 'model.tar.gz')
    with tarfile.open(model_file) as tar:
        for f in files:
            tar.getmember(f)
