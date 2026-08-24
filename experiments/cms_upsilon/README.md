# CMS upsilon peaks in the dimuon channel

This experiment reconstructs and plots the three bottomonium resonances

\[
\Upsilon(1S),\quad \Upsilon(2S),\quad \Upsilon(3S)
\]

in the CMS dimuon (`mu+ mu-`) channel. The default input is the small
[CMS Open Data `Ymumu.csv` sample](https://opendata.cern.ch/record/5206):
20,000 opposite-sign dimuon candidates recorded in 2011 with
`8 < m(mumu) < 12 GeV`. The code recomputes each invariant mass from the two
stored muon four-vectors.

The sample is intended for education and outreach, not a publication-quality
CMS measurement.

## Run

From the repository root, using the CMS Python environment:

```powershell
conda activate cms
python experiments/cms_upsilon/plot_upsilon.py
```

On the first run, the script downloads the 2.6 MB CSV file into the ignored
`experiments/cms_upsilon/data/` directory. It writes:

- `figures/cms_upsilon_mumu.png`: data and the three-peak fit
- `figures/cms_upsilon_mumu.pdf`: vector version of the same plot
- `figures/cms_upsilon_fit_results.csv`: fitted peak positions and resolutions

Useful options:

```powershell
# Plot without fitting
python experiments/cms_upsilon/plot_upsilon.py --no-fit

# Use a local copy of the CMS CSV file
python experiments/cms_upsilon/plot_upsilon.py --input path/to/Ymumu.csv

# Use the larger 2012 reduced NanoAOD ROOT sample already used by cms_zpeak
python experiments/cms_upsilon/plot_upsilon.py --input path/to/Run2012BC_DoubleMuParked_Muons.root
```

The ROOT input path uses exactly-two-muon, opposite-charge events, matching the
official CMS outreach dimuon-spectrum example. ROOT files are processed in
chunks, so the full 2.1 GiB sample does not need to fit in memory.

Run the small reconstruction and input-format checks with:

```powershell
python experiments/cms_upsilon/test_plot_upsilon.py
```

## Model

The figure follows the project's existing CMS resonance style: an 8 x 5.5
single-panel step histogram, Matplotlib's standard blue/orange/green sequence,
an unboxed legend, and the same title, label, and grid treatment as the nearby
CMS Z-peak plots.

The displayed curve is a descriptive fit, not an official CMS result. It uses
three Gaussian detector-resolution peaks plus a second-order exponential
background. The Gaussian means start near the Particle Data Group masses and
are allowed to move within narrow windows. The natural widths of the upsilon
states are much smaller than the CMS dimuon mass resolution in this sample.

Data source: T. McCauley, *Y to two muons from 2011*, CERN Open Data Portal,
record 5206. The data are CC0. Neither CMS nor CERN endorses this analysis.
