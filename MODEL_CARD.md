## Pretrained Models
Follow the link to download the pretrained models.

| dataset     | config                             | checkpoint                                                                                               | F1@50 | Acc  | 
|-------------|------------------------------------|----------------------------------------------------------------------------------------------------------|-------|------|
| Breakfast   | configs/Breakfast/LTContext.yaml   | [`link`](https://drive.filen.io/f/95a780bd-31ac-4496-b32b-543545b24e06#lAhkCP158sA38OkR7FKHsy8fU2kz7kgp) | 61.9  | 74.6 |
| Assembly101 | configs/Assembly101/LTContext.yaml | [`link`](https://drive.filen.io/d/d66d7a7f-0071-468e-b3e0-2948558ce58b#DrrnC7YKqaubpzTVsBYb0wqDMP2V7UtT) | 23.2  | 41.6 |


## Testing the model

Here is an example of the command for testing the model with the pretrained weights downloaded from above links. 
```bash
python run_net.py \
  --cfg configs/Assembly101/LTContext.yaml \
  DATA.PATH_TO_DATA_DIR [path_to_your_dataset] \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.CHECKPOINT_PATH [path_to_checkpoint]
```
For more options look at the TEST section of `ltc/config/defaults.py`.
Note that for Breakfast dataset you need to set `DATA.CV_SPLIT_NUM` to specify the cross-validation split number. 