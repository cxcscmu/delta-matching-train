# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the delta-matching-train authors, 2026.

import os
from copy import deepcopy

from utils.io import load


__all__ = [
    "resolve_and_load_config",
    "partial_update_config",
    "build_config",
]


def resolve_and_load_config(path: str, config_name: str = None) -> dict:
    path = os.path.realpath(os.path.expanduser(path))
    if config_name is not None:
        path = os.path.join(path, config_name)
    config = load(path)
    return config


def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        if key in config and isinstance(partial_config[key], dict) and isinstance(config[key], dict):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]
    return config


def build_config(config: str | dict, recursive=True) -> dict:
    """Build a configuration dictionary from a YAML file (or a dict). A key `include: <path>` loads that file as
    the base and applies this config on top. A missing file raises FileNotFoundError."""
    if isinstance(config, str):
        if not os.path.isfile(config):
            raise FileNotFoundError(f"Config file {config} does not exist.")
        default_config = resolve_and_load_config(config)
        if not isinstance(default_config, dict):
            raise ValueError(f"Config file {config} must hold a YAML mapping, got {type(default_config).__name__}.")
    else:
        if not isinstance(config, dict):
            raise ValueError("Config must be a string path or a dictionary")
        default_config = config

    config = deepcopy(default_config)
    if recursive:
        for k in default_config:
            if isinstance(default_config[k], dict):
                _config = build_config(default_config[k], recursive=True)
                config[k] = _config

        if "include" in config:
            include_path = config.pop("include")
            _config = build_config(include_path, recursive=True)
            config = partial_update_config(_config, config)

    return config
