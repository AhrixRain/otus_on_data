# configs_sota/

Run E/F and Run G0 configuration candidates.

- `cms_Jpsi_ptj5_runE_tierA.yaml` is the intended Run E full configuration.
  It is additive on Run D and enables all Tier A components except the
  Monge-gap regularizer (kept off for the first probe).
- `cms_Jpsi_ptj5_runF_flowmatch.yaml` is the completed independent
  bidirectional flow-matching ablation.
- `cms_Jpsi_ptj5_runG0_reciprocal_bridge.yaml` is the locked-contract shared
  reciprocal-field prototype. It adds worst-direction checkpoint selection
  and writes the G0 manifest/evaluation automatically.

Launch policy: `--dry-run` first, `--smoke` for code-path checks, and do not
start the full 60+100+140 epoch run without explicit authorization.
