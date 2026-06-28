Vendored DLA Components
=======================

These files are vendored from:

https://github.com/aiAdrian/flatland-ecml2026-starterkit/tree/main/experimental/flatland_solver/policy/dla/vendor

They are used as a safety supervisor around the RL/Rerank policy. The source
repository is MIT licensed; see `LICENSE` in this directory.

Local changes:

- Matplotlib import is lazy so normal submission inference does not initialize
  plotting.
