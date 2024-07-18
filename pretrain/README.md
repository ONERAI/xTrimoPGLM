# xTrimoPGLM 

Set `DATA_PATH`, `MULTITASK_DATA_PATH`, `CHECKPOINT_PATH` in `configs/xtpglm-100b/100b.sh` and `HOST_FILE_PATH` in `scripts/submit_gpu.sh`. Run the following scripts to reproduce xtpglm-100b's  training.

```
bash scripts/submit_gpu.sh configs/xtpglm-100b/100b.sh
```

At least 24 DGX-A100 (40G) is needed to lanuch training. A more detailed README will be released soon.
