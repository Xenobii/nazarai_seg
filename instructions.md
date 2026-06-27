# Segmentation dataset evaluation

I want to perform a training study on a semantic segmentation  dataset. The dataset is under /dataset. I want the following:

- I want to train 3 standard semantic segmentation models: classic UNet, DeepLabV3+ and PSPNet. Import them as simply as possible. Only train the final 3 layers.
- I want to use a 70/30 split of the dataset with five-fold cross validation.
- Use pytorch + hydra + pytorch_lightning + tensorboard.
- Configure various configurable hyperparameters using hydra + optuna to get to a good model.
- User miOU as the evaluation metric. Use standard semantic segmentation loss functions.
- Add augmentations for scale, rotation and flip. Images and masks are non-standard dimensions and in 4K resolution, rescale accordingly (equivalent hd or full hd if it doesn't destroy my 4060).
- Run at most 30 epochs per run and at most 20 runs per model. Define a standard batch size which doesn't bloat my 4060 too much.

## End Goal

- A table containing the best performance (mIoU) of each model and the configured hyperparameters for that score. 
- The saved checkpoints of each of these models.
- A folder containing inference results for the entire dataset per model.

## Testing
- Ask for my input when something is unclear. I highly encourage it.
- Treat the project as a quick study - not a proper project.
- Avoid assertions, warnings etc.