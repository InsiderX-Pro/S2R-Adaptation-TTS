# Research website

The website is served by GitHub Pages from `main:/docs` at
https://insiderx-pro.github.io/S2R-Adaptation-TTS/.
The repository remains private; only the contents of `docs/` are deployed.
Do not change Pages to serve the repository root: it contains the private training source.

## Content provenance

- The five-page paper PDF is synchronized byte-for-byte with `Template.pdf` from
  [manuscript commit `02302f1`](https://github.com/piedpiperG/ICASSP2027-SythenticTTS-Luoji/commit/02302f1f46228d5c5ee8cb8526ac594d77cc0959)
  on 23 September 2026. Its SHA-256 is
  `6bd090e6ffb6f6d07855f49793e8247be083b0065f0512970f4bba0bbac64481`.
  The website contribution cards retain the latest manuscript claims; the data
  protocol and reliability validation include real-recording screening and the
  3.2% identical-but-incorrect normalized-transcript result on the full Burmese
  FLEURS training split. The separate 2,921-utterance correlation and CER results
  remain intact. Page 5 now includes an Ethical Statement before the references,
  addressing speaker identity, impersonation, authorization, and source terms.
  The abstract, title and seven-author list are unchanged.
- Figure 1 is rendered from the second page of the previously supplied manuscript
  synchronized on 20 September 2026; the latest manuscript retains the same diagram.
- Tables 1–3, three-seed sample standard deviations, MOS confidence intervals,
  independent-ASR scores and experimental descriptions follow that manuscript.
- The 11 selected WAV files, their metadata and the download archive are preserved
  from the existing website. There is one selected text per language: Burmese
  case13 and Lao case12. These clips illustrate the method and are not a new sample
  of the revised listening evaluation.
- Audio selection used the highest cubic overall score in the existing listening
  collection. That overall score averages naturalness, similarity and pronunciation;
  it differs from the paper's naturalness-only MOS.
- On 15 September 2026 the author corrected the selected S→R audio labels from
  uniform 0.5 to cubic. The correction affects these clip labels, not the paper's
  separate uniform-weight ablations. Original audio bytes remain unchanged.
- The GitHub resource links to this private source repository. Hugging Face hosts
  the public OmniVoice Burmese `cubic` native adapter and inference helpers.
  Paper results are reported separately and are not benchmark claims for this download.
- The paper is available as [arXiv:2609.25951](https://arxiv.org/abs/2609.25951)
  (v1, submitted 22 September 2026). The website and repository PDF remain the
  later 23 September manuscript with the Ethical Statement. No accepted venue is
  claimed.

## Editing and preview

- `docs/index.html`: content, external baselines, ablations and citation.
- `docs/app.js`: Table 1 results, tabs, audio playback and citation copy.
- `docs/style.css`: typography and responsive styling.
- `docs/assets/paper.pdf` and `docs/assets/method.png`: manuscript and figure.
- `docs/assets/demo-data.json` and `docs/demo-data.js`: matching audio metadata.

From the repository root, run `python -m http.server 8765 --directory docs` and open
http://localhost:8765/. No frontend build is required. Native audio controls,
single-player playback, keyboard language/result tabs and citation copy are retained.

When updating the paper, revise all tables and descriptions together. Verify
all three result tabs, narrow-screen horizontal table scrolling, audio resources
and the PDF before pushing `main`; then check the GitHub Pages deployment.
