# data/

## `example_net.xml`

Despite the filename, this is a REACLIB-derived **production network**,
not a tutorial file. It's the `example_net.xml` shipped with the
[wnnet examples](https://github.com/mbradle/wnnet), and turns out to
carry the full JINA REACLIB rate catalog with AME2011 nuclide masses.

### What it is
Production-grade webnucleo network XML. Generated from JINA REACLIB
via the `jina_to_webnucleo` pipeline (see
[this blog post](https://sourceforge.net/u/mbradle/blog/2018/12/downloading-various-webnucleo-network-xml-files/)
for details of the canonical `my_net.xml` → `my_net_full.xml` family).

### Full content
- **1,084 nuclides** (max Z = 60, max A = 150).
- **6,679 reactions** total.
- File size: 22.6 MB.

### Rate sources (top 11, covering 99% of reactions)
| source | count | notes |
| --- | --- | --- |
| `ths8` | 4,634 | Thielemann/Rauscher Hauser-Feshbach statistical model (2008) |
| `wc12` | 747 | Wiescher/Cyburt weak rates (2012) |
| `rath` | 623 | Rauscher theoretical rates |
| `bqa+` | 213 | |
| `ka02` | 160 | |
| `il10` | 48 | Iliadis proton-capture compilation (2010) |
| `cf88` | 32 | Caughlan & Fowler (1988) |
| `rpsm` | 28 | |
| `nacr` | 28 | NACRE compilation |
| `nfis` | 25 | neutron-induced fission |
| `bhi+` | 23 | |

Plus 60 minor-source labels. Full breakdown available via:
```bash
python -c "from collections import Counter; import wnnet.net as w; \
  net = w.Net('data/example_net.xml'); \
  c = Counter(r.source.strip() for r in net.get_valid_reactions().values()); \
  [print(f'{s!r:10s} : {n}') for s, n in c.most_common()]"
```

### Nuclide source
Every nuclide has `<source>ame11</source>`: AME2011 atomic mass
evaluation.

### Variant
Forward-only. The file contains no `<reverse>` tags; wnnet computes
reverse reaction rates at runtime via detailed balance from the
forward rates and partition functions. This is the
`my_net.xml`-style variant from the webnucleo blog post, not
`my_net_full.xml`.

### Project filters
Two filters are used throughout the codebase:

| filter | nuclides | reactions | strong+EM | weak |
| --- | --- | --- | --- | --- |
| narrow `[z <= 8 and a <= 20]` | 30 | 97 | 82 | 15 |
| wide `[z <= 20 and a <= 50]` | 154 | 816 | 676 | 140 |

The narrow filter is the default (fast iteration during CRAM-16
development, equilibrium preservation test). The wide filter is used
for scaling verification and will be used for the Radau accuracy
comparison.

### Filename note
The file is kept as `example_net.xml` for argparse-default stability
across the scripts in this repo. Treat `example_net.xml` as the
canonical REACLIB snapshot for this work; do not rename it.
