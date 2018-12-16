# Sketch-RNN-GAN
GAN fasion of Sketch RNN

Please take the following steps to run our model:

1. To train Sketch-RNN-GAN, run "python train_gan.py". Models will be saved in saved_models/ every 100 epochs. One image will be generated every 100 epochs and saved in output_images/.
2. To train Sketch-RNN baseline model, run "python train_rnn.py". Models will be saved in saved_models/. One image will be generated every 100 epochs and saved in output_images/.
3. To generate image with saved models, run "python generate_with_pretrained_model.py". Generated images will be in output_images. 
4. To train Sketch-RNN-GAN with tau experiment, run "python train_experiment_tau.py"
5. To train SKetch-RNN-GAN with adversarial weight experiment, run "python train_experiment_adv.py".
6. To modify hyperparameters, modify hyperparameter value in hyperparameters.py.


We have placed one dataset (bicycle) in dataset/. To run with additional datasets, download from https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn and put *.npz in dataset/.


