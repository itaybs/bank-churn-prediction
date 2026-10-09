# Dashboard screenshots

The main README shows these images once they exist. To add them:

1. Run `python main.py` once (the tuning page needs its saved models).
2. Start the dashboard: `streamlit run app.py --server.address=localhost --server.port=8501`.
3. Set the browser window to about 1440 px wide. Light or dark theme both work; use the same one for every image.
4. Capture each view below and save it in this folder under the given file name (PNG).
5. In the main `README.md`, remove the `<!--` and `-->` around the matching image line.

| File name | Section | What to capture |
|---|---|---|
| `01-executive-briefing.png` | 0. Executive Briefing & Journey Map | Top of the page: hero banner and the three mission cards. |
| `02-journey-map.png` | 0. Executive Briefing & Journey Map | Churn-trigger cards and the journey map with a stage selected. |
| `03-model-comparison.png` | 3. Model Explorer & Comparison | The *ROC curves* tab. |
| `04-tuning-before-after.png` | 4. Hyperparameter Tuning | The KPI row and the *Before vs after* chart. |
| `05-cross-validation.png` | 5. Cross-Validation & Robustness | The box plot of fold scores. |
| `06-prediction-tool.png` | 7. Interactive Churn Prediction Tool | A high-risk profile (e.g. age 55, inactive, 3 products, Germany) showing the gauge and the driver chart. |
| `07-story-game-board.png` | 9. The Chief Retention Officer | A week in progress: persona cards with chat bubbles, the model switcher, threshold and HUD. |
| `08-story-game-debrief.png` | 9. The Chief Retention Officer | After *Reveal*: the debrief with the score ring, KPIs and lessons. |

Keep each image under about 1 MB so the README loads quickly (crop to the relevant area).
