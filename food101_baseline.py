from baseline_methods.train_keras_model import train_keras_baseline
import argparse


def main(dest_root,
         model_name,
         train_img_dir,
         val_img_dir,
         test_img_dir,
         n_epoch,
         optimzier='nadam',
         init_weight=None,
         batch_size=128):
    trained_weights = train_keras_baseline(expr_root_dir=dest_root,
                                           train_img_dir=train_img_dir,
                                           val_img_dir=val_img_dir,
                                           test_img_dir=test_img_dir,
                                           n_epochs=n_epoch,
                                           optimzier=optimzier,
                                           init_model_weight=init_weight,
                                           which_model=model_name,
                                           batch_size=batch_size)
    return trained_weights


def args_params():
    args = argparse.ArgumentParser()
    args.add_argument('--dest-root', type=str,required=True)
    args.add_argument('--n-epoch',type=int, required=True)
    args.add_argument('--train-img-dir',type=str, required=True)
    args.add_argument('--test-img-dir', type=str,required=True)
    args.add_argument('--optimizer',type=str, required=True)
    args.add_argument('--val-img-dir', type=str,required=True)

    return args

if __name__ == '__main__':
    params = args_params()
    ar = params.parse_args()

    dest_root = ar.dest_root
    n_epoch = ar.n_epoch
    optimzer = ar.optimizer
    train_img_dir = ar.train_img_dir#'/l/workspace/dataset/asghar/train/images'
    val_img_dir = ar.val_img_dir#'/l/workspace/dataset/asghar/val/images'
    test_img_dir = ar.test_img_dir# '/l/workspace/dataset/asghar/test/images'
    last_trained_weights = {'MobileNet': (None, 512), 'Xception': (None, 6)}
    for q in range(10):
        for m in last_trained_weights.keys():
            new_w = main(dest_root=dest_root,
                         model_name=m,
                         train_img_dir=train_img_dir,
                         val_img_dir=val_img_dir,
                         test_img_dir=val_img_dir,
                         n_epoch=n_epoch,
                         optimzier=optimzer,
                         init_weight=last_trained_weights[m][0],
                         batch_size=last_trained_weights[m][1])

            last_trained_weights[m] = (str(new_w), last_trained_weights[m][1])
