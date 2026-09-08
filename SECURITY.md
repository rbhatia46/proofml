# Security and data handling

ProofML's built-in checks read local tabular data, document strings, and retrieval
IDs/judgments. They do not execute
notebooks, deserialize pickle models, call network services, or modify inputs.
Reports contain column names, aggregates, configuration, and source hashes;
these can still be sensitive. Store reports according to your data policy.

Third-party checks run with the Python process's privileges and are trusted
code. The engine's exception isolation is not a security sandbox. Plugins are
responsible for omitting raw values. Avoid processing concurrently changing
inputs. The current memory/byte caps are not a hardened resource sandbox.

Report exploitable issues privately using the repository's GitHub security
advisory facility once enabled by the maintainer. Do not attach private datasets
or credentials to public issues.
