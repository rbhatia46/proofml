# What generalizes, and what does not

ProofML checks are independent of business-domain column names. Customer churn,
prices, energy demand, and sensor tables use the same algorithms. Reading a
table successfully is different from proving that its ML evaluation is valid.

| Input/workflow | Current support | What you supply |
| --- | --- | --- |
| UTF-8 comma-separated CSV | Yes, scalar cells, unique nonempty headers | Path; optional target |
| Scalar Parquet | Yes, optional pandas/pyarrow extra | Path |
| pandas 2.x DataFrame | Yes, numeric/string/category/datetime/nullable scalars | Frame with named columns; materialize relevant index |
| Binary/multiclass classification | Common checks; Pearson association only for binary numeric labels | Target and optional test split |
| Numeric regression | Common checks, numeric-label validation | `task="regression"`, target |
| Single/multiple forecast series | Fixed horizons or per-row label availability at one shared cutoff | Forecast origin, target, optional series and availability timestamps/horizon |
| Unlabelled test data | Supported | Test predictor columns; target health only assessed where labels exist |
| Data only, no target | Quality/split checks run; target checks skip | Input table(s) |
| Large datasets beyond configured limits | Rejected; engine is in-memory | Reduce data deliberately or use an out-of-core system |
| Dense NumPy arrays / separate X and y | Supported through pandas extra | 2D features, 1D labels with matching length; generated array column names |
| Sparse matrices | Not directly supported | Deliberately convert a bounded subset to a named DataFrame |
| Polars, Spark, SQL handles, Excel | No native adapter | Explicitly export/convert to a supported format |
| Nested JSON, lists in cells, images, text semantics | Not supported as model-specific analyses | Separate domain checks/tools |
| Business-day/monthly cadence, rolling retraining | Not fully supported | Separate fold contracts/custom checks; variable label availability itself is supported |

## Normalization and limits

CSV uses comma separators, UTF-8 (BOM accepted), and stripped cell strings.
Empty or whitespace-only CSV cells are missing; literal `NA`/`NULL` are not
guessed to be missing. pandas/Parquet nulls become empty strings. Nonfinite
numeric strings are not treated as valid numeric regression labels.
Values normalize as strings: `1` and `1.0` can differ across heterogeneous
sources. Standardize dtypes when comparing file/frame splits. The adapter
does not infer units, parse locale-specific numbers, or reconcile category aliases.

Default caps are 200,000 rows and 100 MB per input. File byte caps refer to
on-disk bytes; DataFrame caps use pandas deep memory estimates including the
index. Actual audit memory can be higher. DataFrame fingerprints cover normalized
columns/values; file fingerprints cover source bytes. Cross-format hashes are
not meant to match. Index and dtype changes alone need not change frame hashes.

## Detection limits

Unusual correlation does not prove leakage. Identical rows may be legitimate
repeated measurements. Missing labels may be expected in inference data.
The tool cannot know whether a feature was available when a real prediction was
made unless you declare that fact. It does not inspect model code or fit order,
evaluate model performance, certify fairness, or guarantee absence of leakage.

The tests validate implementation contracts and synthetic failure/clean cases.
They do not measure real-world recall or false-positive rates across all domains.
Use it as a reproducible first-pass review and CI contract tool, alongside
domain-specific validation. A universal “safe dataset” claim would be misleading.
