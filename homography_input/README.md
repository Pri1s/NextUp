# Homography test-frame input

Place one or more image frames here (`.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`).

Run the classical CV pipeline against this folder:

```bash
source .venv/bin/activate
python run_homography_labels.py homography_input --output homography_results
```

This folder is only an input location for homography testing. It is not used by
the dataset assembly or learned-model labeling workflows.
