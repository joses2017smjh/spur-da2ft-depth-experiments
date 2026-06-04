# Transferring the 88 GB dataset

The image/depth data is **not** in this repo (too large for git). It lives on the
OSU HPC cluster and must be copied to wherever the experiments will run.

- **Source (OSU HPC):** `/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/Data/`
- **Login host:** `submit.hpc.engr.oregonstate.edu`
- **What to copy:** only the box-family `bark_brown_02` subset — **26,520 files /
  87.66 GB** — listed in [`data_transfer_filelist.txt`](data_transfer_filelist.txt)
  (paths are relative to the source `Data/` dir, e.g.
  `full_spur/depth/bark_brown_02/...`).

After transfer you must end up with this layout, and point `DATA_ROOT` at the dir
that **contains** `full_spur/`:
```
<DATA_ROOT>/full_spur/{depth,Da2Finetune,Optical_flow,ann,mask}/bark_brown_02/<tree>/<box*>/...
```

`rsync` (not `scp`) is required — `--files-from` selects just the 88 GB subset out
of a much larger tree. `--partial` makes it resumable: if the link drops, re-run
the exact same command and it continues.

---

## Fastest path: friend pulls straight to his HPC node (HPC → HPC)
If your friend's HPC can SSH to OSU, skip the laptop entirely. **On his node:**
```bash
# grab the file list (it's in this repo, or scp it from OSU)
scp sanchej7@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/sanchej7/spur_data_filelist.txt .

rsync -a --partial --info=progress2 \
  --files-from=data_transfer_filelist.txt \
  sanchej7@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/Data/ \
  /path/to/DATA_ROOT/
```

---

## Via a Windows laptop (no WSL needed)

### 1. Pull OSU → laptop
WSL isn't required. Use a tool that bundles `ssh`/`rsync`:

**MobaXterm** (portable, no admin) — https://mobaxterm.mobatek.net/download-home-edition.html
Open *Start local terminal*. Note: `C:` is `/drives/c` in MobaXterm (NOT `/mnt/c`,
which is WSL-only):
```bash
scp sanchej7@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/sanchej7/spur_data_filelist.txt .

rsync -a --partial --info=progress2 \
  --files-from=spur_data_filelist.txt \
  sanchej7@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/Data/ \
  /drives/c/Users/sanchej7/Computer_Vision/Data/
```
(In **Git Bash** instead, `C:` is `/c/...`. Git Bash lacks `rsync` by default —
MobaXterm is simpler.)

Requirements on the laptop: ~90 GB free on `C:`.

### 2. Send laptop → friend's HPC
Now it's already filtered, so just mirror the local folder (no file list needed):
```bash
rsync -a --partial --info=progress2 \
  /drives/c/Users/sanchej7/Computer_Vision/Data/ \
  friend@his-hpc-host:/path/to/DATA_ROOT/
```
(or have your friend pull from the laptop, or use any file-transfer he prefers).

---

## Via Google Drive (zip)
Works if you'd rather hand off a single file than give your friend SSH access.
Drive's free tier is 15 GB, so 88 GB needs **Google One (100 GB ≈ $2/mo)**.

1. **Get the data onto the laptop** first (the MobaXterm rsync above).
2. **Zip it** (one file uploads/downloads far better than 26,520 loose files).
   On Windows use **7-Zip** → right-click the `Data` folder → *7-Zip → Add to
   archive*. `.npy`/`.png` barely compress, so pick **Store** (fast) and
   optionally **Split to volumes, 10 GB** for a resumable multi-part upload.
   Result: `Data.zip` (or `Data.zip.001`, `.002`, …).
3. **Upload** `Data.zip` to Google Drive (web drive.google.com, or the Drive
   desktop app). Right-click → *Share* → "Anyone with the link" → copy the link.
4. **Friend downloads on his HPC** with `gdown` (no Drive app needed on a node):
   ```bash
   pip install --user gdown
   gdown --fuzzy "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing" -O Data.zip
   unzip Data.zip -d /path/to/DATA_ROOT_parent   # yields .../full_spur/...
   ```
   For split volumes, download each `.001/.002/...` then
   `7z x Data.zip.001` (or `zip -s 0 Data.zip --out joined.zip && unzip joined.zip`).

> Heads-up: a public Drive file downloaded many times can hit Google's
> "quota exceeded" wall. `gdown --fuzzy` handles the big-file confirm token; if
> you hit the quota, share to your friend's Google account directly instead of a
> public link, or use `rclone copy` with his own Drive auth.

---

## Then run
On the machine with an NVIDIA GPU + Python 3.10:
```bash
DATA_ROOT=/path/to/DATA_ROOT sbatch --partition=<his> run_spur_cnn_fusion_RGBD_NOcalib_da2ft_seeds.sh
DATA_ROOT=/path/to/DATA_ROOT sbatch --partition=<his> run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh
```
See [README.md](README.md) for full setup.
