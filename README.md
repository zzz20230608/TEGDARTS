# Try to use TEGNAS in DARTS

TEGNAS: Understanding and Accelerating Neural Architecture Search with Training-Free and Theory-Grounded Metrics [[PDF](https://arxiv.org/pdf/2108.11939.pdf)]

#### 1. Search
```python
python train_search.py  --gpu 0
```
### 2. Train
```python
python train.py --cutout --auxiliary --gpu 0
```
