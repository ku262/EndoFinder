# Reproduce evaluation results

## Polyp-Twin evaluations

We run `EndoFinder/polyp_eval.py` to evaluate the `EndoFinder` model
using our default preprocessing (resize to 224, preserving the aspect ratio).

```bash
EndoFinder/polyp_eval.py --hash=True --codecs=Flat --model=vit_large_patch16 \
  --output_path=/path/to/eval/output \
  --model_state=/path/to/EndoFinder.pth \
  --polyp_path=/path/to/Polyp-Twin \
  --size=224 --preserve_aspect_ratio=true 
```

The command produces a CSV file,
`polyp_metrics.csv`, in the configured `--output_path`:

```
codec,score_norm,uAP,accuracy-at-1,recall-at-p90
Flat,None,0.6932980827229672,0.693069306930693,0.5247524752475248
```

If we don't use the hash layer in the end, we need to change the hash in 
the command to False, and the result is as follows:

```
codec,score_norm,uAP,accuracy-at-1,recall-at-p90
Flat,None,0.695408017737841,0.693069306930693,0.49504950495049505
```


