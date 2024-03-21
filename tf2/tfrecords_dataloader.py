import tensorflow_datasets as tfds

features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(224, 224, 3)),
        'label': tfds.features.ClassLabel(names=['fossil']),
        })

split_infos = [
    tfds.core.SplitInfo(
        name='train',
        shard_lengths=[1000],  # Num of examples in shard0, shard1,...
        num_bytes=0,  # Total size of your dataset (if unknown, set to 0),
    ),
]

tfds.folder_dataset.write_metadata(
    data_dir='/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/data1/',
    features=features,
    # Pass the `out_dir` argument of compute_split_info (see section above)
    # You can also explicitly pass a list of `tfds.core.SplitInfo`.
    split_infos=split_infos,
    # Pass a custom file name template or use None for the default TFDS
    # file name template.
    filename_template=None,

    # Optionally, additional DatasetInfo metadata can be provided
    # See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    #description="""Multi-line description."""
    #homepage='http://my-project.org',
    supervised_keys=('image', 'label'),
    #citation="""BibTex citation.""",
)

builder = tfds.builder_from_directory('/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/data1/')

ds = builder.as_dataset(split='train', shuffle_files=True)

import matplotlib.pyplot as plt

for x in ds.take(1):
    plt.imshow(x['image'])
    plt.show()
    break