# CS180 Project 3 — Fun with Flow Models

This project contains the completed Part A and Part B starter notebooks, reusable Python implementations, and a monochrome results website populated with the exported experiment images. The website keeps experiment images in color.

## Project layout

```text
.
├── notebooks/
│   ├── CS180_Project3_PartA_Final.ipynb
│   └── CS180_Project3_PartB_Final.ipynb
├── src/flow_models/                 # Reusable Part A samplers and MNIST models
├── scripts/
│   ├── complete_uploaded_notebooks.py
│   ├── build_website.py
│   └── train_mnist.py               # Optional local Part B runner
├── assets/results/                  # Filled by build_website.py after Colab runs
└── index.html                       # GitHub Pages entry point, ready to publish
```

## Run the notebooks in Colab

1. Download each `.ipynb` from `notebooks/`.
2. In Google Colab, choose **File → Upload notebook** and upload one notebook.
3. Choose **Runtime → Change runtime type → T4 GPU**.
4. Run **Runtime → Run all**. Each notebook has its own setup cell and embedded course support, so no project ZIP or Drive mount is needed.
5. At the end, download the generated `parta-website-results.zip` and `partb-website-results.zip` files. Part A loads PixNerd and takes substantially longer than MNIST training. For your final submission, use `GROUP='all'` and `QUICK_RUN=False`.

The Part A notebook defaults to the course Campanile image and two simple generated drawings so it can run without input uploads. Edit the image paths/prompts in the labeled cells to use your own photos and drawings. The Part B notebook runs the required 5-epoch denoisers, 10-epoch flow models, guided samples, and learning-rate comparison.

## Build the website from notebook results

Put both downloaded ZIP files in the project folder, then run:

```bash
python scripts/build_website.py --parta parta-website-results.zip --partb partb-website-results.zip
```

This copies the report figures into `assets/results/parta/` and `assets/results/partb/`, where `index.html` loads them. The current project already includes the result ZIPs you provided. Preview locally with:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`. For the website PDF deliverable, use the browser's **Print → Save as PDF** after confirming the images load.

## Publish with GitHub Pages

1. Create a new GitHub repository (for example, `CS180-Project3`).
2. Put the project files at the repository root, including `index.html` and `assets/results/`.
3. Commit and push the files to the `main` branch.
4. In GitHub, open **Settings → Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/(root)`, and save.
5. After deployment, the project page URL will be `https://YOUR_USERNAME.github.io/CS180-Project3/`. Replace `YOUR_USERNAME` and the repository name with yours. Check the Pages deployment status if the page is not available immediately.

## Local code

Install the dependencies in `requirements.txt` to use the local Part A helpers and Part B runner. Part A's pretrained PixNerd weights are loaded by the course-provided Colab setup and are not bundled in this repository.
