## Pretrained Models
Follow the link to download the pretrained models.

| dataset     | config                             | checkpoint                                                                                                                                                                                                                                                                                                                                                                                       | F1@50 | Acc  | 
|-------------|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|------|
| Breakfast   | configs/Breakfast/LTContext.yaml   | [`sp1`](https://bit.ly/42KpExR), [`sp2`](https://bit.ly/3wwc7xC), [`sp3`](https://bit.ly/49CVyhQ), [`sp4`](https://bit.ly/3uGLHbY) | 61.9  | 74.6 |
| Assembly101 | configs/Assembly101/LTContext.yaml | [`link`](https://www.dropbox.com/scl/fi/epsabc9yfivkyiy5vwfc5/assembly101_ltc_ckpt.pth?rlkey=phzw65rl3rbns33ue5fk1253y&dl=0)                                                                                                                                                                                                                                                                     | 23.2  | 41.6 |


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