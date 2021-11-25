# SAGCN - Official PyTorch Implementation
#### | [Paper](https://www.sciencedirect.com/science/article/pii/S0893608020304512) | [Project Page](https://github.com/tczhangzhi/PyTorch-SAGCN)
This is the official implementation of the paper "Steganographer detection via a similarity accumulation graph convolutional network". NOTE: We are refactoring this project to the best practice of engineering.

## Abstract

Steganographer detection aims to identify guilty users who conceal secret information in a number of images for the purpose of covert communication in social networks. Existing steganographer detection methods focus on designing discriminative features but do not explore relationship between image features or effectively represent users based on features. In these methods, each image is recognized as an equivalent, and each user is regarded as the distribution of all images shared by the corresponding user. However, the nuances of guilty users and innocent users are difficult to recognize with this flattened method. In this paper, the steganographer detection task is formulated as a multiple-instance learning problem in which each user is considered to be a bag, and the shared images are multiple instances in the bag. Specifically, we propose a similarity accumulation graph convolutional network to represent each user as a complete weighted graph, in which each node corresponds to features extracted from an image and the weight of an edge is the similarity between each pair of images. The constructed unit in the network can take advantage of the relationships between instances so that common patterns of positive instances can be enhanced via similarity accumulations. Instead of operating on a fixed original graph, we propose a novel strategy for reconstructing and pooling graphs based on node features to iteratively operate multiple convolutions. This strategy can effectively address oversmoothing problems that render nodes indistinguishable although they share different instance-level labels. Compared with the state-of-the-art method and other representative graph-based models, the proposed framework demonstrates its effectiveness and reliability ability across image domains, even in the context of large-scale social media scenarios. Moreover, the experimental results also indicate that the proposed network can be generalized to other multiple-instance learning problems.

## Roadmap
After many rounds of revision, the project code implementation is not elegant. Thus, in order to help the readers to reproduce the experimental results of this paper quickly, we will open-source our study following this roadmap:

- [x] refactor and open-source all the model files, training files, and test files of the proposed method for comparison experiments.
- [ ] refactor and open-source the visualization experiments.
- [ ] refactor and open-source the APIs for the real-world steganographer detection in an out-of-box fashion.

## Quick Start

#### Dataset and Pre-processing

We use the MDNNSD model to extract a 320-D feature from each image and save the extracted features in different `.mat` files. You should check `./data/train` and `./data/test` to confirm you have the dataset ready before experiments. For example, `cover.mat` and `suniward_01.mat` should be placed in the `./data/train` and `./data/test` folders.

Then, we provide a dataset tool to distribute image features and construct innocent users and guilty users as described in the paper, for example:

```
python preprocess_dataset.py --target suniward_01_100 --guilty_file suniward_01 --is_train --is_test --is_reset --mixin_num 0
```

#### Train the proposed SAGCN

To obtain our designed model for detecting steganographers, we provide an entry file with flexible command-line options, arguments  to train the proposed SAGCN on the desired dataset under various experiment settings, for example:

```
python main.py --epochs 80 --batch_size 100 --model_name SAGCN --folder_name suniward_01_100 --parameters_name=sagcn_suniward_01_100 --mode train --learning_rate 1e-2 --gpu 1
python main.py --epochs 80 --batch_size 100 --model_name SAGCN --folder_name suniward_01_100 --parameters_name=sagcn_suniward_01_100 --mode train --learning_rate 1e-2 --gpu 1
```

#### Test the proposed SAGCN

For reproducing the reported experimental results, you just need to pass command-line options of the corresponding experimental setting, such as:

```
python main.py --batch_size 100 --model_name SAGCN --parameters_name sagcn_suniward_01_100 --folder_name suniward_01_100 --mode test --gpu 1
```

#### Visualize

If you set `summary` to `True` during training, you can use `tensorboard` to visualize the training process.

```
tensorboard --logdir logs --host 0.0.0.0 --port 8088
```

## Requirement

- Hardware: GPUs Tesla V100-PCIE (our version)
- Software:
  - h5py==2.7.1 (our version)
  - scipy==1.1.0 (our version)
  - tqdm==4.25.0 (our version)
  - numpy==1.14.3 (our version)
  - torch==0.4.1 (our version)

## Contact

If you have any questions, please feel free to open an issue.

## Contribution

We thank all the people who already contributed to this project:

* Zhi ZHANG
* Mingjie ZHENG
* Shenghua ZHONG
* Yan LIU

## Citation Information

If you find the project useful, please cite:

```
@article{zhang2021steganographer,
  title={Steganographer detection via a similarity accumulation graph convolutional network},
  author={Zhang, Zhi and Zheng, Mingjie and Zhong, Sheng-hua and Liu, Yan},
  journal={Neural Networks},
  volume={136},
  pages={97--111},
  year={2021}
}
```