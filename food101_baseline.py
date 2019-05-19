from baseline_methods.train_keras_model import train_keras_baseline


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


if __name__ == '__main__':
    dest_root = 'keras_baseline'
    n_epoch = 10
    optimzer = 'nadam'
    train_img_dir = '/l/workspace/dataset/asghar/train/images'
    val_img_dir = '/l/workspace/dataset/asghar/val/images'
    test_img_dir = '/l/workspace/dataset/asghar/test/images'
    last_trained_weights = {'MobileNet': ('/l/workspace/deep_project/keras_baseline/MobileNet/0/top_model_weights.h5', 512),'Xception': ('/l/workspace/deep_project/keras_baseline/Xception/0/top_model_weights.h5', 6)}
    for q in range(10):
        for m in last_trained_weights.keys():
            new_w = main(dest_root=dest_root,
                         model_name=m,
                         train_img_dir=train_img_dir,
                         val_img_dir=val_img_dir,
                         test_img_dir=val_img_dir,
                         n_epoch=n_epoch,
                         optimzier='nadam',
                         init_weight=last_trained_weights[m][0],
                         batch_size=last_trained_weights[m][1])

            last_trained_weights[m] = (str(new_w), last_trained_weights[m][1])
