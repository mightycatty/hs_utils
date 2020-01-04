# assert float(tklib.__version__) >= 1.14, 'tensorflow version of over 1.14 is required for fully functional apis'
import os
import warnings
import tensorflow as tf
# disable warnings
warnings.filterwarnings("ignore")
# disable tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


def model_wrapper(model_fn):
    input_tensor = tf.keras.layers.Input(InputFnConfig.shape, name='input')
    segmentation_output = model_fn(input_tensor)
    model = tf.keras.models.Model(input_tensor, segmentation_output, name=ModelConfig.model_name)
    return model


def get_latest_checkpoint(folder):
    file_list = os.listdir(folder)
    latest_file = max(file_list, key=os.path.getctime)
    return latest_file


def main():
    # fire configuration via telegram bot
    message = 'ModelName:{}\n\n' \
              'InputConfig:\nTrainRecord:{}\nBS:{}\nAugment:{}\nShape:{}\n\n' \
              'TrainingConfig:\nCheckpoint:{}\nOptimizer:{}\nlr:{}'\
        .format(
                ModelConfig.model_name,
                InputFnConfig.train_record_path, InputFnConfig.batch_size, InputFnConfig.augmentation_flag, InputFnConfig.shape,
                TrainingConfig.checkpoint, TrainingConfig.optimizer, TrainingConfig.lr)
    try:
        ModelConfig.logger.fire_message_via_bot(message)
    except Exception as e:
        print ('bot not working:{}'.format(e))
    if not LINUX_PLATFORM_FLAG:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
    tf.keras.backend.set_image_data_format('channels_last')

    train_dataset = InputFnConfig.train_dataset
    val_dataset = InputFnConfig.val_dataset
    model = model_wrapper(ModelConfig.model_fn)
    # freeze specific layers
    if TrainingConfig.not_frozen_layers is not None:
        for layer_item in model.layers:
            if layer_item.name in TrainingConfig.not_frozen_layers:
                layer_item.trainable = True
            else:
                layer_item.trainable = False
                print ('freeze layer:{}'.format(layer_item))
    for layer_item in model.layers:
        if 'BN' in layer_item.name:
            if TrainingConfig.freeze_bn:
                layer_item.trainable = False
                print ('freeze BN:{}'.format(layer_item))
    # 强行load weights, 利用except来处理多GPU和单gpu weight问题
    try:
        model.load_weights(TrainingConfig.checkpoint, by_name=True)
        print ('load weights from: {}'.format(TrainingConfig.checkpoint))
    except Exception as e:
        print ('unable to load weights-single-gpu')
        print (e)
        pass
    if TrainingConfig.GPU_num > 1:
        model = multi_gpu_model(model, gpus=TrainingConfig.GPU_num, cpu_merge=False, cpu_relocation=False)
        try:
            if TrainingConfig.checkpoint == 'latest':
                checkpoint_dir = os.path.join(LOG_ABS_DIR, 'saved_checkpoints', ModelConfig.model_name)
                cp = get_latest_checkpoint(checkpoint_dir)
                model.load_weights(cp, by_name=True)
            else:
                model.load_weights(TrainingConfig.checkpoint, by_name=True)
            print ('load weights from: {}'.format(TrainingConfig.checkpoint))
        except Exception as e:
            print('unable to load weights-multi-gpu')
            print (e)
            pass
    # compile model
    model.compile(optimizer=TrainingConfig.optimizer,
                  loss=ModelConfig.loss,
                  metrics=ModelConfig.metrics,
                  )
    # create callbacks
    callback = create_callbacks(ModelConfig.model_name, LOG_ABS_DIR)
    model.fit(train_dataset,
              epochs=TrainingConfig.epochs,
              steps_per_epoch=TrainingConfig.steps_per_epoch,
              callbacks=callback,
              validation_data=val_dataset,
              validation_steps=InputFnConfig.val_step,
              verbose=1,
              initial_epoch=TrainingConfig.initial_epoch,
              )


if __name__ == '__main__':
    main()
