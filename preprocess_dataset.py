#!/usr/bin/env python
# coding=utf-8

'Construct datasets of users based on extrated image features (from .mat)'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import argparse

from dataset_tool import SocialDatasetTool, TestDatasetTool, TrainDatasetTool

parse = argparse.ArgumentParser()
parse.add_argument(
    "--target", help="Name of target file", default='suniward_04_100')
parse.add_argument(
    "--guilty_file", help="Name of guilty file", default='suniward_04')
parse.add_argument(
    "--normal_file", help="Name of normal file", default='cover')
parse.add_argument(
    "--mixin_num", help="Number of normal image mixed in guilty user", type=int, default=0)
parse.add_argument(
    "--is_train", help="Generate train and val dataset or not", action='store_true')
parse.add_argument(
    "--is_test", help="Generate test dataset or not", action='store_true')
parse.add_argument(
    "--is_case", help="Generate case dataset or not", action='store_true')
parse.add_argument(
    "--is_social", help="Generate social dataset or not", action='store_true')
parse.add_argument(
    "--is_reset", help="Reset the folder or not", action='store_true')
parse.add_argument(
    "--social_payload", help="Payload of social guilty file", default='04')
parse.add_argument(
    "--target_num", help="Reset the folder or not", type=int, default=1)
args = parse.parse_args()

target = args.target
guilty_file = args.guilty_file
normal_file = args.normal_file
mixin_num = args.mixin_num
is_train = args.is_train
is_test = args.is_test
is_case = args.is_case
is_social = args.is_social
is_reset = args.is_reset
target_num = args.target_num
social_payload = args.social_payload

print("Setting:",
    f"target={target}",
    f"guilty_file={guilty_file}",
    f"normal_file={normal_file}",
    f"mixin_num={mixin_num}",
    f"is_train={is_train}",
    f"is_test={is_test}",
    f"is_case={is_case}",
    f"is_social={is_social}",
    f"is_reset={is_reset}",
    f"target_num={target_num}",
    f"social_payload={social_payload}")

# Replace current results
if is_reset:
    TrainDatasetTool.reset_target_folder(target_folder=target)

# For training datasets of spatial/frequency domain experiments
if is_train:
    train_generator = TrainDatasetTool()
    train_generator.save(target_folder=target, normal_images_file=normal_file, guilty_images_file=guilty_file, mixin_num=mixin_num)

# For test datasets of spatial/frequency domain experiments
if is_test:
    test_generator = TestDatasetTool()
    test_generator.save(target_folder=target, normal_images_file=normal_file, guilty_images_file=guilty_file, mixin_num=mixin_num)

# For social network experiments
if is_social:
    social_generator = SocialDatasetTool(payload=social_payload)
    social_generator.save(target_folder=target, target_num=target_num)
