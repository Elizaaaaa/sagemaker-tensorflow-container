# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os

import numpy as np
<<<<<<< HEAD
import pytest
from sagemaker.tensorflow import serving, TensorFlow

from test.integration import RESOURCE_PATH
from test.integration.utils import processor, py_version  # noqa: F401
=======
from sagemaker.tensorflow import serving, TensorFlow

from test.integration import RESOURCE_PATH
>>>>>>> Add Keras support (#126)


logging.basicConfig(level=logging.DEBUG)


<<<<<<< HEAD
@pytest.mark.skip(reason="Serving part fails because of version mismatch.")
def test_keras_training(sagemaker_local_session, docker_image, tmpdir, framework_version):
=======
def test_keras_training(sagemaker_local_session, docker_image, tmpdir):
>>>>>>> Add Keras support (#126)
    entry_point = os.path.join(RESOURCE_PATH, 'keras_inception.py')
    output_path = 'file://{}'.format(tmpdir)

    estimator = TensorFlow(
        entry_point=entry_point,
        role='SageMakerRole',
        train_instance_count=1,
        train_instance_type='local',
<<<<<<< HEAD
<<<<<<< HEAD
        image_name=docker_image,
        sagemaker_session=sagemaker_local_session,
        model_dir='/opt/ml/model',
        output_path=output_path,
        framework_version=framework_version,
=======
=======
        image_name=docker_image,
>>>>>>> Create parameter server in different thread (#129)
        sagemaker_session=sagemaker_local_session,
        model_dir='/opt/ml/model',
        output_path=output_path,
        framework_version='1.11.0',
>>>>>>> Add Keras support (#126)
        py_version='py3')

    estimator.fit()

<<<<<<< HEAD
<<<<<<< HEAD
    model = serving.Model(model_data=output_path,
                          role='SageMakerRole',
                          framework_version=framework_version,
=======
    model = serving.Model(model_data=output_path, role='SageMakerRole',
=======
    model = serving.Model(model_data=output_path,
                          role='SageMakerRole',
>>>>>>> Create parameter server in different thread (#129)
                          framework_version='1.11.0',
>>>>>>> Add Keras support (#126)
                          sagemaker_session=sagemaker_local_session)

    predictor = model.deploy(initial_instance_count=1, instance_type='local')

    assert predictor.predict(np.random.randn(4, 4, 4, 2) * 255)

    predictor.delete_endpoint()
