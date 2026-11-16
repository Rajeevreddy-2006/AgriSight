# Knowledge base

The assistant answers treatment questions **only** from documents in this folder.
The three `example_*.md` files are short placeholders written in general terms so
the pipeline runs end to end. **They are not official advice and contain no doses.**

## Replace them with official sources

Download crop protection guidelines (PDF is fine, they are parsed page by page) from
sources such as:

- ICAR institutes and the ICAR crop-specific advisories
- State agricultural universities (e.g. TNAU Agritech Portal, PJTSAU, ANGRAU, OUAT)
- State agriculture department package-of-practices documents
- Central Insecticides Board & Registration Committee (CIB&RC) approved-use lists
- FAO Integrated Pest Management guides

Put the files here, then rebuild the index:

```bash
python -m agrisight.rag.index
```

Keep a `SOURCES.md` listing where each file came from and its date. Mention in your
report that the knowledge base was curated, and from which sources.
