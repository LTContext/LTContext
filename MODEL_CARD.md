## Pretrained Models
Follow the link to download the pretrained models.

| dataset     | config                             | checkpoint                                                                                       | F1@50 | Acc  | 
|-------------|------------------------------------|--------------------------------------------------------------------------------------------------|-------|------|
| Breakfast   | configs/Breakfast/LTContext.yaml   | [`link`](https://drive.google.com/drive/folders/1pjLtPlPKN0DfJfr1TRDwQfzRF5gJSz_0?usp=sharing)   | 61.9  | 74.6 |
| Assembly101 | configs/Assembly101/LTContext.yaml | [`link`](https://drive.google.com/file/d/1t1Kfqld4kdrfH9PP8kckgs8jA7daBrbD/view?usp=sharing)     | 23.2  | 41.6 |


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