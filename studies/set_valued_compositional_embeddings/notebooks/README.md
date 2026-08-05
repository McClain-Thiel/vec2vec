# Interpretation Notebooks

Create notebooks only after a versioned Kedro reporting artifact exists.

Use this order:

```text
00_overview.ipynb
10_constraint_audit.ipynb
20_split_and_gallery_audit.ipynb
30_set_supervision_results.ipynb
40_composition_results.ipynb
```

Create only the notebooks that the completed work needs. Each notebook must begin with its
conclusion and exact input artifact identifiers. Keep metric computation in tested library or
pipeline code. Use notebooks to inspect, explain, and visualize persisted results.
