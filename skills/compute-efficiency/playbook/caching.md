# Caching features and preprocessing (slow_startup, uncached_features)

**Signal.** Long time before the first step, or embeddings and features recomputed on every run.

**Fix. Does not change results when the cache key covers everything that affects the output.**
```python
key = hashlib.sha1(f"{model_name}|{data_path}|{data_mtime}|{max_len}".encode()).hexdigest()[:12]
path = Path(cache_dir) / f"features-{key}.npy"
if path.exists():
    feats = np.load(path)
else:
    feats = extract(model, data)
    np.save(path, feats)
```
Put the cache on scratch or project storage, not in `$HOME` if the quota is small. Invalidate it when the model, tokenizer, data version or preprocessing parameters change.

**Verify.** Check that cached and freshly computed features match (`np.allclose`) on a subset once.
