# app.py
import os, re, uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- plug your logic here ----
from mail_logic import analyze  # make sure analyze(files_dict, reporter=...) RETURNS a dict

from PIL import Image, ImageDraw


# ==============================
# Streamlit Reporter (for live status)
# ==============================
class StReporter:
    """Minimal reporter used by analyze()/create_group_pdfs."""
    def __init__(self):
        self.status = {}   # stage -> st.status
        self.progress = {} # stage -> (bar, total, done)

    def start(self, stage, total=None, label=None):
        box = st.status(label or stage, expanded=True)
        self.status[stage] = box
        if total:
            self.progress[stage] = [st.progress(0.0, text=label or stage), int(total), 0]

    def tick(self, stage, inc=1, message=None):
        if stage in self.progress:
            bar, total, done = self.progress[stage]
            done = min(total, done + (inc or 1))
            self.progress[stage][2] = done
            bar.progress(done/total, text=message or f"{done}/{total}")
        if stage in self.status and message:
            self.status[stage].write(message)

    def log(self, stage, message):
        if stage in self.status and message:
            self.status[stage].write(message)

    def done(self, stage, success=True, message=None):
        if stage in self.status:
            self.status[stage].update(
                label=f"{'✅' if success else '❌'} {message or stage}",
                state="complete" if success else "error"
            )


# ==============================
# Config
# ==============================
st.set_page_config(page_title="PDF Duplicate Finder", page_icon="📄", layout="wide")
UPLOAD_ROOT = "uploads"
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# ==============================
# Session State
# ==============================
if "files" not in st.session_state:
    # files: dict[file_id] -> {id, name, size_bytes, uploaded_at, paths{base,pages,ocr_text,thumbnails,original}}
    st.session_state.files = {}
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False
if "error" not in st.session_state:
    st.session_state.error = None
# store analysis result
if "analysis_out" not in st.session_state:
    st.session_state.analysis_out = None


# ==============================
# Utils
# ==============================
def fmt_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}"
        size /= 1024


def _safe_stem(filename: str) -> str:
    """Make a safe, lowercase folder name from filename (without extension)."""
    stem = Path(filename).stem.lower().strip()
    stem = re.sub(r"\s+", "_", stem)             # spaces -> underscore
    stem = re.sub(r"[^a-z0-9._-]", "", stem)     # keep only safe chars
    return stem or "file"


def file_dirs_from_name(fname: str, root: str = "uploads", create: bool = True):
    """
    Returns (base, pages_dir, ocr_dir, thumbs_dir, original_dir).
    Uses the SAME safe stem everywhere.
    NEVER auto-uniques; caller decides whether to skip or reuse.
    """
    stem = _safe_stem(fname)
    base = os.path.join(root, stem)
    pages_dir    = os.path.join(base, "pages")
    ocr_dir      = os.path.join(base, "ocr_text")
    thumbs_dir   = os.path.join(base, "thumbnails")
    original_dir = os.path.join(base, "original")
    if create:
        os.makedirs(pages_dir, exist_ok=True)
        os.makedirs(ocr_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
    return base, pages_dir, ocr_dir, thumbs_dir, original_dir


def add_files(uploaded_files, root: str = "uploads"):
    st.session_state.setdefault("files", {})
    added, skipped = [], []

    for f in uploaded_files:
        buf = f.getbuffer()
        size_bytes = getattr(f, "size", None) or len(buf)

        # compute paths (no creation yet)
        base, pages_dir, ocr_dir, thumbs_dir, original_dir = file_dirs_from_name(
            f.name, root=root, create=False
        )
        original_path = os.path.join(original_dir, f.name)

        if os.path.isdir(base):
            st.info(f"'{os.path.basename(base)}' already exists. Using existing.")
            os.makedirs(original_dir, exist_ok=True)

            if os.path.exists(original_path):
                # re-register if missing in session
                if not any(m["paths"]["base"] == base for m in st.session_state.files.values()):
                    fid = str(uuid.uuid4())
                    st.session_state.files[fid] = {
                        "id": fid,
                        "name": f.name,
                        "size_bytes": size_bytes,
                        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                        "paths": {
                            "base": base,
                            "pages": pages_dir,
                            "ocr_text": ocr_dir,
                            "thumbnails": thumbs_dir,
                            "original": original_path,
                        },
                    }
                skipped.append(f.name)
                continue
            else:
                # save missing original
                os.makedirs(original_dir, exist_ok=True)
                with open(original_path, "wb") as out:
                    out.write(buf)
        else:
            # create structure and save original
            base, pages_dir, ocr_dir, thumbs_dir, original_dir = file_dirs_from_name(
                f.name, root=root, create=True
            )
            original_path = os.path.join(original_dir, f.name)
            with open(original_path, "wb") as out:
                out.write(buf)

        fid = str(uuid.uuid4())
        st.session_state.files[fid] = {
            "id": fid,
            "name": f.name,
            "size_bytes": size_bytes,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "base": base,
                "pages": pages_dir,
                "ocr_text": ocr_dir,
                "thumbnails": thumbs_dir,
                "original": original_path,
            },
        }
        added.append(f.name)

    if added:
        st.success(f"✅ Added {len(added)} file(s): {', '.join(added)}")
    if skipped:
        st.info(f"⏭️ Skipped {len(skipped)} existing: {', '.join(skipped)}")


def remove_file(file_id: str):
    st.session_state.files.pop(file_id, None)


def clear_all():
    st.session_state.files.clear()
    st.session_state.is_analyzing = False
    st.session_state.error = None
    st.session_state.analysis_out = None


# ==============================
# Results Renderer (no analysis here)
# ==============================
def render_results(out):
    if not out or "results" not in out or not out["results"]:
        st.warning("No grouping results returned.")
        return

    results = out["results"]  # per file_id
    file_name_by_id = {f["id"]: f["name"] for f in st.session_state.get("files", {}).values()}

    rows = []
    total_pages = 0
    total_groups = 0

    for file_id, blob in results.items():
        fname = file_name_by_id.get(file_id, file_id)

        # blob["groups"] may be either:
        # 1) {"groups": {...}, "threshold": ..., ...}
        # 2) directly the map {"group_1": {...}, ...}
        raw = (blob or {}).get("groups", {}) or {}

        # Pick the inner groups dict
        if isinstance(raw, dict) and "groups" in raw and isinstance(raw["groups"], dict):
            inner_groups = raw["groups"]
        elif isinstance(raw, dict):
            inner_groups = raw
        else:
            inner_groups = {}

        total_groups += len(inner_groups)

        for gname, gdata in inner_groups.items():
            members = gdata.get("members", [])
            size = gdata.get("size", len(members))
            total_pages += size
            rows.append({
                "File": fname,
                "Group": gname,
                "Size": size,
                "Representative": gdata.get("rep", ""),
                "Similarity": gdata.get("similarity", 0.0),
            })

    if not rows:
        st.warning("No groups found in results.")
        return

    df = pd.DataFrame(rows).sort_values(["File", "Group"])

    st.title("📑 Grouping Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Groups", total_groups)
    c2.metric("Total Pages in Groups", total_pages)
    # If your analyze() returns "threshold", show it; else placeholder
    threshold = out.get("threshold", "—")
    c3.metric("Threshold", threshold)
    c4.metric("Max Neighbors", "—")

    st.subheader("📋 Groups Table")
    st.dataframe(df[["File", "Group", "Size", "Representative", "Similarity"]],
                 use_container_width=True)

    st.subheader("📂 Details")
    # Expanders grouped by file for readability
    for fname in df["File"].unique():
        sub = df[df["File"] == fname]
        st.markdown(f"### 📄 {fname}")
        # resolve file_id for this fname
        file_id = next((fid for fid, n in {f["id"]: f["name"] for f in st.session_state.get("files", {}).values()}.items() if n == fname), None)
        for _, r in sub.iterrows():
            g = {}
            if file_id and file_id in results:
                g = results[file_id].get("groups", {}).get(r["Group"], {})
            with st.expander(f"{r['Group']} • {int(r['Size'])} items • {r['Similarity']}% similarity"):
                st.write(f"**Representative:** {g.get('rep','')}")
                members = g.get("members", [])
                if members:
                    st.write("**Members:**")
                    st.code("\n".join(members))

    st.subheader("📈 Charts")
    st.caption("Group Size by File/Group")
    df_chart = df.copy()
    df_chart["Label"] = df_chart["File"] + " • " + df_chart["Group"]
    st.bar_chart(df_chart.set_index("Label")["Size"])


# ==============================
# Sidebar (controls)
# ==============================
with st.sidebar:
    st.header("📄 PDF Duplicate Finder")
    st.caption("Manage files & run analysis")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_b:
        if st.button("🧹 Clear All", type="secondary"):
            clear_all()
            st.rerun()

    st.markdown("---")

    st.subheader("Upload PDFs")
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        add_files(uploaded)

    st.subheader("Uploaded Files")
    if not st.session_state.get("files"):
        st.info("No files yet.")
    else:
        for f in st.session_state.files.values():
            with st.container(border=True):
                st.write(f"📄 **{f['name']}**")
                st.caption(f"{fmt_size(f['size_bytes'])} • {f['uploaded_at']}")
                if st.button("Remove", key=f"rm_{f['id']}", type="secondary"):
                    remove_file(f["id"])
                    st.rerun()

    st.markdown("---")

    # Downloads sourced from generated folders
    with st.expander("⬇️ Downloads (group_pdfs)", expanded=False):
        files_map = st.session_state.get("files", {})
        if not files_map:
            st.info("No files found.")
        else:
            for file_id, meta in files_map.items():
                pages_dir = Path(meta["paths"]["pages"])
                group_dir = pages_dir / "group_pdfs"
                if not group_dir.exists():
                    continue
                pdfs = sorted(group_dir.glob("*.pdf"))
                if not pdfs:
                    continue

                st.markdown(f"**{meta['name']}** — {len(pdfs)} PDF(s)")
                for pdf in pdfs:
                    try:
                        data_bytes = pdf.read_bytes()
                        st.download_button(
                            label=pdf.name,
                            data=data_bytes,
                            file_name=pdf.name,
                            mime="application/pdf",
                            key=f"dl-{file_id}-{pdf.name}",
                        )
                    except Exception as e:
                        st.caption(f"⚠️ Could not read {pdf.name}: {e}")

    st.markdown("---")
    run = st.button("🔎 Analyze", type="primary", use_container_width=True)
    if run:
        st.session_state.is_analyzing = True
        st.session_state.analysis_out = None
        st.rerun()


# ==============================
# Main: loading vs results
# ==============================
if st.session_state.get("is_analyzing"):
    rep = StReporter()
    # If your analyze signature accepts reporter:
    out = analyze(st.session_state.get("files", {}), reporter=rep)
    # If not yet updated, you could fallback to: out = analyze(st.session_state.get("files", {}))
    st.session_state.analysis_out = out
    st.session_state.is_analyzing = False
    st.rerun()

# Render results if present
if st.session_state.get("analysis_out"):
    render_results(st.session_state["analysis_out"])


# ==============================
# Footer
# ==============================
st.divider()
st.caption("UI-only shell. Plug in your analyzer to populate groups/thresholds.")


# def _reporter_self_test():
#     rep = StReporter()
#     rep.start("render", total=5, label="Rendering pages (self-test)")
#     for i in range(5):
#         import time; time.sleep(0.3)
#         rep.tick("render", message=f"Page {i+1}/5")
#     rep.done("render", message="Render complete")
#
# st.button("▶️ Reporter Self-Test", on_click=_reporter_self_test)
