# cur-prot

All the scripts used in curation of records should live here.

New records are curated by running the following from the root of the project (one dir up from `scripts`:

```
RiksdagenCorpus $  python scripts/src/cur-prot/pipeline.py
RiksdagenCorpus $  python scripts/src/cur-prot/post-pipeline.py
```

Important: pay attention to the required args and/or environment variables. Run each script with `-h` to see.

Alternatively, look at the curation steps in post-pipeline and run them one at a time.