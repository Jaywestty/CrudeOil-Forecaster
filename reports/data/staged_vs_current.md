# Data diff: `data` → `data\staging`

## oil_macro_weekly.csv

- Old: 1044 rows, 2006-01-08 → 2026-01-04
- New: 1089 rows, 2006-01-08 → 2026-11-15
- Dates added: 45 (2026-01-11 → 2026-11-15)
- Dates removed: 0

| column | shared rows | rows changed | max abs change | first changed | last changed |
|---|---|---|---|---|---|
| brent | 1044 | 8 | 2.5500 | 2006-01-15 | 2026-01-04 |
| inventories | 1044 | 1 | 3587.0000 | 2026-01-04 | 2026-01-04 |
| industrial_prod | 1044 | 18 | 0.5908 | 2025-09-07 | 2026-01-04 |
| dollar_index | 1044 | 1 | 0.0010 | 2026-01-04 | 2026-01-04 |
| fed_funds | 1044 | 1 | 0.0800 | 2026-01-04 | 2026-01-04 |
| vix | 1044 | 1 | 0.0042 | 2026-01-04 | 2026-01-04 |

## oil_macro_transformed.csv

- Old: 1043 rows, 2006-01-15 → 2026-01-04
- New: 1088 rows, 2006-01-15 → 2026-11-15
- Dates added: 45 (2026-01-11 → 2026-11-15)
- Dates removed: 0

| column | shared rows | rows changed | max abs change | first changed | last changed |
|---|---|---|---|---|---|
| brent_return | 1043 | 14 | 0.0410 | 2006-01-15 | 2026-01-04 |
| dollar_return | 1043 | 1 | 0.0000 | 2026-01-04 | 2026-01-04 |
| indpro_return | 1043 | 5 | 0.0045 | 2025-09-07 | 2026-01-04 |
| inventory_pct | 1043 | 1 | 0.0043 | 2026-01-04 | 2026-01-04 |
| fed_funds_diff | 1043 | 1 | 0.0800 | 2026-01-04 | 2026-01-04 |
| vix_diff | 1043 | 1 | 0.0042 | 2026-01-04 | 2026-01-04 |
