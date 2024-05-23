import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import pandas as pd
import model as model_lib
from absl import flags
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_util import preprocess_for_eval


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('dataset', None, 'Dataset name')
flags.DEFINE_string('train_mode', 'linear_evaluation', 'Training mode.')
flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')
flags.DEFINE_integer('width_multiplier', 1, 'Width multiplier to scale number of channels.')
flags.DEFINE_integer('image_size', 224, 'Input image size.')
flags.DEFINE_float('sk_ratio', 0., 'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')
flags.DEFINE_boolean('global_bn', True, 'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')
flags.DEFINE_float('se_ratio', 0., 'If it is bigger than 0, it will enable SE.')
flags.DEFINE_integer('proj_out_dim', 128, 'Number of head projection dimension.')
flags.DEFINE_enum('proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'], 'How the head projection is done.')
flags.DEFINE_integer('num_proj_layers', 3, 'Number of non-linear head layers.')
flags.DEFINE_bool('lineareval_while_pretraining', True, 'Whether to finetune supervised head while pretraining.')
flags.DEFINE_boolean('use_blur', True, 'Whether or not to use Gaussian blur for augmentation during pretraining.')
flags.DEFINE_integer('ft_proj_selector', 0, 'Which layer of the projection head to use during fine-tuning. ''0 means no projection head, and -1 means the final layer.')
flags.DEFINE_string('path_to_imagefolder', "/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20", 'Path to the image folder')
flags.DEFINE_string('destination', "test_linear_eval", 'Path to the destination folder')
FLAGS(sys.argv)

if __name__ == "__main__":
    
    print("Loading data...")
    
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        flags.FLAGS.path_to_imagefolder,
        image_size=[flags.FLAGS.image_size, flags.FLAGS.image_size],
        shuffle=False,
    )
    ds = ds.unbatch()
    ds = ds.map(lambda x, y: (x / 255., y))
    ds = ds.map(lambda x, y: (preprocess_for_eval(x, 224, 224), y))
    ds = ds.batch(32)
        
    print("Loading encoder...")
    
    # Load the model from the checkpoint
    checkpoint_dir = "/Users/ima029/Desktop/simclr-v2/simclr-v2/tf2/trained_models/7111501"
    model = model_lib.Model(num_classes=0)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    os.makedirs(flags.FLAGS.destination, exist_ok=True)
    
    print("Extracting features...")
        
    X = []
    y = []
    
    for x, z in ds:
        feats = model(x, training=False)[0]
        X.append(feats)
        y.append(z)
    
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    summary_tables = [pd.DataFrame() for _ in range(10)]
    
    for seed in range(10):
        print(f"Evaluating seed {seed}...")
        # create a train and test split
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        log_model = LogisticRegression(
            random_state=seed,
            max_iter=10000,
            multi_class="multinomial",
            class_weight="balanced",
        )
    
        log_model.fit(X_tr, y_tr)
        
        y_pred = log_model.predict(X_te)
        
        summary_tables[seed].loc["logistic", "log_loss"] = log_loss(y_te, log_model.predict_proba(X_te))
        summary_tables[seed].loc["logistic", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
        summary_tables[seed].loc["logistic", "accuracy"] = accuracy_score(y_te, y_pred)
        summary_tables[seed].loc["logistic", "mean_precision"] = precision_score(y_te, y_pred, average="macro")
    
        # fit a knn model for each k
        for k in range(1, 10, 2):
            
            knn = KNeighborsClassifier(n_neighbors=k, p=2)
            knn.fit(X_tr, y_tr)
            y_pred = knn.predict(X_te)
            
            # compute the accuracy and balanced accuracy and mean precision on the test set
            summary_tables[seed].loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(y_te, y_pred)
            summary_tables[seed].loc[f"k={k}", "accuracy"] = accuracy_score(y_te, y_pred)
            summary_tables[seed].loc[f"k={k}", "mean_precision"] = precision_score(y_te, y_pred, average="macro")
                
    mean_summary_table = pd.concat(summary_tables).groupby(level=0).mean()
    mean_summary_table.to_csv(os.path.join(flags.FLAGS.destination, "mean_summary_table.csv"))
    for i, summary_table in enumerate(summary_tables):
        summary_table.to_csv(os.path.join(flags.FLAGS.destination, f"summary_table_seed{i}.csv"))
