Reproduction order (single CPU, about 25 minutes in total):
  python experiments.py 2 10      # round 1 (H1-H4), writes raw_D2.csv, raw_D10.csv
  python experiments2.py 2        # round 2 (H5-H8), writes raw2_D2.csv
  python experiments2.py 10       # round 2, writes raw2_D10.csv
  python logging_exp.py           # realised-information diagnostics, writes logging_out.json
  python analysis2.py             # tables (t*.tex), tests, facts.json, fig_sweep2.pdf
  python fig_info.py              # fig_info.pdf
Paper: paper_final.tex (needs IEEEtran.cls, fig_info.pdf, fig_sweep2.pdf). The t*.tex table bodies produced by
analysis2.py were pasted into the placeholders of the LaTeX source.
Older drafts and scripts are in archive/.
