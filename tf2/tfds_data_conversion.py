import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# hyperparameters
data_dir = '/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/tfds'
image_shape = (224, 224, 3)
shard_size = 96080

# Create a FeaturesDict corresponding to the dataset that has been written in tfrecords format
features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=image_shape),
        'label': tfds.features.ClassLabel(names=['fossil']),
        })

# Necessary metadata to write the dataset
split_infos = [
    tfds.core.SplitInfo(
        name='train',
        shard_lengths=[shard_size],  # Num of examples in shard0, shard1,...
        num_bytes=0,  # Total size of your dataset (if unknown, set to 0),
        num_shards=32,  # Num of shards in the split
    ),
]

tfds.folder_dataset.write_metadata(
    data_dir=data_dir,
    features=features,
    split_infos=split_infos,
    # Pass a custom file name template or use None for the default TFDS
    # file name template.
    filename_template=None,
    supervised_keys=('image', 'label'),
    # Optionally, additional DatasetInfo metadata can be provided
    # See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    # description="""Multi-line description."""
    # homepage='http://my-project.org',
    # citation="""BibTex citation.""",
)

# test the dataset
builder = tfds.builder_from_directory(data_dir)


builder = tfds.folder_dataset.ImageFolder


ds = builder.as_dataset(split='train', shuffle_files=True)

# count the number of examples
i = 0
for x in ds:
    i += 1

for x in ds.take(1):
    plt.imshow(x['image'])
    plt.show()
    break

    builder = tfds.builder('imagenet2012')


ds = ds.map(lambda x: preprocess_image(x, 224, 224, is_training=True))

import data as data_lib

strategy = tf.distribute.MirroredStrategy()

ds = data_lib.build_distributed_dataset(builder, 32, True, strategy, None)


iterator = iter(ds)

FLAGS = flags.FLAGS


flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'validation',
    'Split for evaluation.')

flags.DEFINE_string(
    'dataset', 'imagenet2012',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', True,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_bool(
    'use_tpu', True,
    'Whether to run on TPU.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')

