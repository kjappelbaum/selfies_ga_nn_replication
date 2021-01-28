# Simultanous logP and QED optimization

In the original code on GitHub no time-adaptive penalty was used and also the SAS was no longer part of the fitness score.

In the original implementation logP and QED also did not receive the same weight, the fitness was calculated base on the centered (i.e., mean subtracted) values

```python
fitness = - (np.square(logP_norm - 2.491233077292206)) - (50 * (np.square(QED_results - 0.9)))
```
