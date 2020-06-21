# A few words
Hi there! :smiley: Like you, I'm an NLP researcher looking for a solution to run GPT-2 project on Tensorflow 2.x. As you may already know, GPT-2 has been developed by OpenAI team and use Tensorflow 1.x as a base framework and unfortunately, Tensorflow 1.x and 2.x are very different and slightly hard to upgrade from 1.x version to 2.x. Tensorflow 2.x removed some libraries / modules that are frequently used in version 1.x such as tf.contrib and also moved / modified a few other ones such as "hparams". In GPT-2 original source code, some parts were broken because of these problems. To use GPT-2 for my private project, I planned to rewrite all source code by Tensorflow 2.0 to fit my old codes and to better understand GPT-2 model. But I realized that my deadlines were killing me :stuck_out_tongue: and in fact, maybe a new implementation could not achieve the best performance of GPT-2. So, I decided to change the GPT-2 source code with some of the smallest possible differences to keep its capabilities. To do it, I cloned a project from a super nice guy: [Awesome GPT-2 with training script](https://github.com/nshepperd/gpt-2) and do some improvements. This is what I have changed:
- Add Hparams class to replace "tf.contrib.training.HParams", you can find it in src/hparams.py file.
- Add "graph_def_editor" module to replace the "graph_editor" module. This is an awesome project written by [CODAIT](https://github.com/CODAIT/). I will place the path of this project in [here](https://github.com/CODAIT/graph_def_editor).
- I have added a new option for the training script of [nshepperd](https://github.com/nshepperd) to choose GPU or CPU device. You can find more details in train.py file
- And a bunch of other minor changes ...

# Testing project
To test this project, you can use pip or whatever you want to set up your environment. If you like me, an Anaconda user :smiley:, I prepared for you an environment file to make your life easier. You can follow these steps below:
1. Install anaconda
2. cd ./src
3. conda env create -f ./environment.yml -p ./.env
4. conda activate ./.env
5. To train and test, you can follow instructions from [nshepperd](https://github.com/nshepperd) project I added below :smiley:. Remember, you can choose which device your model runs on by '--device' flag :stuck_out_tongue:.

# Description of nshepperd and openai projects is below 

Reference:  ["Beginner’s Guide to Retrain GPT-2 (117M) to Generate Custom Text Content"](https://medium.com/@ngwaifoong92/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f)

# gpt-2

Code from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

We have currently released small (117M parameter) and medium (345M parameter) versions of GPT-2.  While we have not released the larger models, we have [released a dataset](https://github.com/openai/gpt-2-output-dataset) for researchers to study their behaviors.

See more details in our [blog post](https://blog.openai.com/better-language-models/).

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2.

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Fine tuning on custom datasets

To retrain GPT-2 117M model on a custom text dataset:

```
PYTHONPATH=src ./train.py --dataset <file|directory|glob>
```

If you want to precompute the dataset's encoding for multiple runs, you can instead use:

```
PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/encoded.npz
PYTHONPATH=src ./train.py --dataset /path/to/encoded.npz
```

Make sure `cudnn` is installed. [Some have reported](https://github.com/nshepperd/gpt-2/issues/8) that `train.py` runs without it but has worse memory usage and might OOM.

### Gradient Checkpointing

https://github.com/openai/gradient-checkpointing is included to reduce the memory requirements of the model, and can be enabled by `--memory_saving_gradients`. The checkpoints are currently chosen manually (poorly) by just adding layer 10 to the 'checkpoints' collection in model.py. `--memory_saving_gradients` is enabled by default for training the 345M model.

### Validation loss

Set `--val_every` to a number of steps `N > 0`, and "validation" loss against a fixed sample of the dataset will be calculated every N steps to get a better sense of training progress. N around 200 suggested. You can set `--val_dataset` to choose a separate validation dataset, otherwise it defaults to a sample from the train dataset (so not a real cross-validation loss!).

### Optimizer

You can use SGD instead of Adam with `--optimizer sgd`. This also helps conserve memory when training the 345M model. Note: the learning rate needs to be adjusted for SGD, due to not having Adam's gradient normalization (0.0006 seems to be a good number from some experiments).

### Multi gpu (out of date)

To do distributed on multiple GPUs or machines using Horovod:

```
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x PYTHONPATH=src \
    -mca pml ob1 -mca btl ^openib \
    /home/jovyan/gpt-2/train-horovod.py --dataset encoded.npz
```

## GPT-2 samples

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

While we have not yet released GPT-2 itself, you can see some samples from it in the `gpt-2-samples` folder.
We show unconditional samples with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.
We show conditional samples, with contexts drawn from `WebText`'s test set, with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

[MIT](./LICENSE)
