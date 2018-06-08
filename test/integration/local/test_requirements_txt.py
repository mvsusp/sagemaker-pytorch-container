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
import os
from test.utils import local_mode
from test.integration import resources_path

source_dir = os.path.join(resources_path, 'source_dir')


def test_mnist_cpu(docker_image, opt_ml, use_gpu):
    local_mode.train('user_script', source_dir, docker_image, opt_ml, use_gpu=use_gpu, source_dir=source_dir)

    assert local_mode.file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'