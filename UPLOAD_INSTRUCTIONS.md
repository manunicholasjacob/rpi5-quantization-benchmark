# GitHub Repository Upload Instructions

## Repository URL
https://github.com/manunicholasjacob/rpi5-quantization-benchmark

## Step-by-Step Upload Process

### 1. Initialize Git Repository (if not already done)

```bash
cd "c:\My files\Windsurf Projects\Paper7 - tier 3\rpi5-quantization-benchmark"
git init
```

### 2. Add Remote Repository

```bash
git remote add origin https://github.com/manunicholasjacob/rpi5-quantization-benchmark.git
```

### 3. Create .gitignore

Create a `.gitignore` file in the repository root with:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# LaTeX
*.aux
*.log
*.out
*.toc
*.synctex.gz

# OS
.DS_Store
Thumbs.db
*.swp
*.swo

# IDE
.vscode/
.idea/
*.sublime-*

# Large model files (if not uploading)
# models/*.onnx
```

### 4. Stage All Files

```bash
git add .
```

### 5. Commit

```bash
git commit -m "Initial commit: INT8 quantization benchmark on Raspberry Pi 5

- Complete paper (3 pages, IEEE ESL format)
- Statistical analysis scripts with 95% CI
- Publication-quality figures with error bars
- Raw experimental data (9,600 measurements)
- Comprehensive README and documentation
- MIT License
- Reproducibility scripts and environment files"
```

### 6. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

## Alternative: Using GitHub Desktop

1. Open GitHub Desktop
2. File → Add Local Repository
3. Choose: `c:\My files\Windsurf Projects\Paper7 - tier 3\rpi5-quantization-benchmark`
4. Publish repository to GitHub
5. Repository name: `rpi5-quantization-benchmark`
6. Description: "Reproducible benchmark for INT8 quantization on Raspberry Pi 5"
7. Keep code private: No (public repository)
8. Click "Publish repository"

## Post-Upload Checklist

### On GitHub.com:

1. **Add repository description**:
   - Go to repository settings
   - Add: "Reproducible benchmark for FP32 vs INT8 inference on Raspberry Pi 5 with ONNX Runtime. Statistical validation with 95% CI, energy analysis, and failure mode identification."

2. **Add topics/tags**:
   - `quantization`
   - `raspberry-pi`
   - `edge-computing`
   - `onnx-runtime`
   - `reproducible-research`
   - `deep-learning`
   - `benchmark`

3. **Enable GitHub Pages** (optional):
   - Settings → Pages
   - Source: Deploy from branch `main`
   - Folder: `/` (root)
   - This will make README.md visible as a website

4. **Create Release** (optional):
   - Go to Releases → Create a new release
   - Tag: `v1.0.0`
   - Title: "Initial Release: INT8 Quantization Benchmark"
   - Description: Include key results and paper citation

5. **Add DOI** (optional):
   - Link repository to Zenodo for permanent DOI
   - Go to https://zenodo.org/
   - Connect GitHub account
   - Enable repository archiving

## Verify Upload

After uploading, verify:

- [ ] README.md displays correctly on main page
- [ ] All folders and files are present
- [ ] Paper PDF opens correctly
- [ ] Scripts are readable
- [ ] Figures display properly
- [ ] LICENSE file is recognized by GitHub
- [ ] Repository is public (if intended)

## Update Paper References

The paper already references the correct GitHub URL:
- `https://github.com/manunicholasjacob/rpi5-quantization-benchmark`

No further updates needed to the paper.

## Troubleshooting

### Large Files
If ONNX models are too large (>100 MB):
- Use Git LFS: `git lfs install`
- Track models: `git lfs track "*.onnx"`
- Or provide download links in `models/README.md`

### Authentication
If prompted for credentials:
- Use Personal Access Token (PAT) instead of password
- Generate at: GitHub Settings → Developer settings → Personal access tokens

### Repository Already Exists
If repository exists on GitHub:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Next Steps

1. Upload repository to GitHub
2. Verify all files are accessible
3. Share repository link in paper submission
4. Consider creating a release for paper publication
5. Add citation information to README if paper is accepted
