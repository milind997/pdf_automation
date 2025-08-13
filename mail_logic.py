

import os, re, json, glob, uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from wget import download

from document_grouping_using_cosine_similarity import group_near_duplicates, summarize_groups
import pandas as pd
import traceback

from pathlib import Path
import io
from typing import Optional


class Reporter:
    """No-op reporter. Replace methods in UI with Streamlit-backed ones."""
    def start(self, stage: str, total: Optional[int] = None, label: Optional[str] = None):
        pass

    def tick(self, stage: str, inc: int = 1, message: Optional[str] = None):
        pass

    def log(self, stage: str, message: str):
        pass

    def done(self, stage: str, success: bool = True, message: Optional[str] = None):
        pass

# convenience: a global that you can pass if you don't care about UI
NULL_REPORTER = Reporter()

















# Optional: use GPU if available (SentenceTransformers supports device param)
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ---------------------
# Config
# ---------------------
UPLOAD_ROOT = "uploads"
INDEX_DIR = os.path.join(UPLOAD_ROOT, "index")  # for saved indices / groups
ALLOWED_EXTENSIONS = {"pdf"}

THUMBNAIL_SIZE = (256, 256)
DPI = 300
SIM_THRESHOLD = 0.90  # 90%



# Ensure folders
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# In-memory state
uploaded_files = {}  # file_id -> {id,name,path,uploadDate,size}
groups_cache = {}  # last computed groups dict

# Load embedding model once
# model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


# ---------------------
# Helpers
# ---------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_dirs(file_id: str):
    """Return per-file directories (ensure created)."""
    base = os.path.join(UPLOAD_ROOT,file_id)
    pages = os.path.join(base, "pages")
    ocr = os.path.join(base, "ocr_text")
    thumbs = os.path.join(base, "thumbnails")

    db_dir = os.path.join(base, "db_dir")


    for d in (base, pages, ocr, thumbs):
        os.makedirs(d, exist_ok=True)
    return base, pages, ocr, thumbs


def page_id_for(file_id: str, page_num: int) -> str:
    return f"{file_id}_p{page_num:04d}"



def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_ocr_texts(ocr_dir: str) -> List[Tuple[str, str]]:
    """
    Read *.txt OCR pages and return [(page_id, text), ...]
    """
    pairs: List[Tuple[str, str]] = []
    for txt_path in sorted(glob.glob(os.path.join(ocr_dir, "*.txt"))):
        page_id = Path(txt_path).stem  # e.g. "page_0001"
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                t = clean(f.read())
        except Exception:
            t = ""
        if t:
            pairs.append((page_id, t))
    return pairs


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Return normalized float32 embeddings (N, D).
    """
    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    return embs


def build_faiss_ip_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an Inner Product index; with normalized vectors this is cosine similarity.
    """
    if embs.ndim != 2 or embs.size == 0:
        raise ValueError("embs must be a non-empty (N, D) array")
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx


def save_artifacts(db_dir: str, idx: faiss.IndexFlatIP, embs: np.ndarray, meta: List[Dict[str, Any]]) -> None:
    """
    Persist FAISS index, embeddings, and metadata.
    """
    base = Path(db_dir)
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "embeddings.npy", embs)
    with open(base / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    faiss.write_index(idx, str(base / "index.faiss"))





def build_vector_db(ocr_dir: str, db_dir: str, reporter: Reporter = NULL_REPORTER) -> Dict[str, Any]:
    pairs = load_ocr_texts(ocr_dir)
    if not pairs:
        reporter.done("vectorize", success=False, message=f"No OCR texts in {ocr_dir}")
        return {"error": f"No OCR texts found in: {ocr_dir}"}

    page_ids, texts = zip(*pairs)
    texts = list(texts)

    reporter.start("vectorize", total=len(texts), label="Embedding & indexing")
    # embed in chunks to drive progress
    batch = 128
    embs_list = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        embs_chunk = embed_texts(chunk)   # your existing function
        embs_list.append(embs_chunk)
        reporter.tick("vectorize", inc=len(chunk), message=f"Embedded {min(i+batch, len(texts))}/{len(texts)}")

    embs = np.vstack(embs_list)

    meta = [{"page_id": pid, "text_path": str(Path(ocr_dir) / f"{pid}.txt")} for pid in page_ids]
    idx = build_faiss_ip_index(embs)
    save_artifacts(db_dir, idx, embs, meta)
    reporter.done("vectorize", success=True, message=f"Indexed {len(texts)} pages")

    return {
        "count": len(texts),
        "vectors_dir": str(Path(db_dir)),
        "index": str(Path(db_dir) / "index.faiss"),
        "meta": str(Path(db_dir) / "meta.json"),
        "embeddings": str(Path(db_dir) / "embeddings.npy"),
    }



def load_vector_db(vectors_dir: str, reporter: Reporter = NULL_REPORTER) -> Dict[str, Any]:
    """
    Load a saved vectors database created by your build_vector_db():
      - index.faiss (IP index)
      - meta.json   (rows metadata)
      - embeddings.npy (optional to load; not required for search)

    Returns a handle with:
      {
        "dir": <Path>,
        "index": <faiss.Index>,
        "meta": <list[dict]>,
        "dim": <int>,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
      }
    """

    reporter.start("load_db", label="Loading vector DB")


    vdir = Path(vectors_dir)
    if not vdir.is_dir():
        raise FileNotFoundError(f"Vectors dir not found: {vdir}")

    index_path = vdir / "index.faiss"
    meta_path  = vdir / "meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.faiss in {vdir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {vdir}")

    index = faiss.read_index(str(index_path))
    meta  = json.loads(meta_path.read_text(encoding="utf-8"))

    # Try to infer dimension (D) from the index; fallback to embeddings.npy if needed
    try:
        dim = index.d
    except Exception:
        emb_path = vdir / "embeddings.npy"
        if not emb_path.exists():
            raise RuntimeError("Cannot infer vector dimension; embeddings.npy also missing.")
        dim = int(np.load(emb_path, mmap_mode="r").shape[1])

    reporter.done("load_db", success=True, message="Vector DB loaded")


    # IMPORTANT: your pipeline normalized vectors; we’ll use IP == cosine
    return {
        "dir": vdir,
        "index": index,
        "meta": meta,
        "dim": dim,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }





def render_pages_if_needed(
    pdf_path: str,
    thumbs_dir: str,
    pages_dir: str,
    dpi: int = 200,
    thumb_size=(320, 320),
    prefix: str = "page_",
    zero_pad: int = 4,
    reporter: Reporter = NULL_REPORTER,
):


    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(thumbs_dir, exist_ok=True)

    written_pages, written_thumbs = [], []

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        reporter.start("render", total=total_pages, label="Rendering pages")
        for idx, page in enumerate(doc, start=1):
            base = f"{prefix}{idx:0{zero_pad}d}"
            png_path   = os.path.join(pages_dir,  f"{base}.png")
            thumb_path = os.path.join(thumbs_dir, f"{base}.thumb.png")

            need_png   = not os.path.exists(png_path)
            need_thumb = not os.path.exists(thumb_path)

            if need_png or need_thumb:
                pix = page.get_pixmap(dpi=dpi)
                if need_png:
                    pix.save(png_path)
                    written_pages.append(png_path)
                if need_thumb:
                    img_bytes = pix.tobytes("png")
                    im = Image.open(io.BytesIO(img_bytes))
                    im.thumbnail(thumb_size)
                    im.save(thumb_path, format="PNG")
                    written_thumbs.append(thumb_path)

                    print("==============dsadsadasds")


            reporter.tick("render", message=f"Page {idx}/{total_pages}")

        reporter.done("render", success=True, message=f"Rendered {len(written_pages)} new page(s)")

    return written_pages, written_thumbs






def ocr_pages_if_needed(pages_dir, ocr_dir, reporter: Reporter = NULL_REPORTER):
    import os
    from PIL import Image
    import pytesseract

    os.makedirs(ocr_dir, exist_ok=True)

    pngs = sorted([f for f in os.listdir(pages_dir) if f.lower().endswith(".png")])
    reporter.start("ocr", total=len(pngs), label="Running OCR")
    for i, fname in enumerate(pngs, start=1):
        pid = os.path.splitext(fname)[0]
        txt_path = os.path.join(ocr_dir, f"{pid}.txt")
        if os.path.exists(txt_path):
            reporter.tick("ocr", message=f"Skipped (cached) {i}/{len(pngs)}")
            continue

        img_path = os.path.join(pages_dir, fname)
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
        except Exception:
            text = ""

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text or "")

        reporter.tick("ocr", message=f"OCR {i}/{len(pngs)}")

    reporter.done("ocr", success=True, message="OCR complete")














def _find_image_for_basename(image_dir: Path, base: str):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
    for ext in exts:
        p = image_dir / f"{base}{ext}"
        if p.exists():
            return p
    # also allow files like base_something.ext, e.g., page_0003-1.jpg
    for p in image_dir.glob(f"{base}*"):
        if p.suffix.lower() in exts:
            return p
    return None

def _load_rgb(path: Path):
    im = Image.open(path)
    # PDF wants RGB; also ensure no alpha
    return im.convert("RGB")






def create_group_pdfs(
    groups_dict: Dict[str, Any],
    image_folder: str,
    output_dir: str,
    *,
    reporter: Optional[object] = None,       # pass your Streamlit reporter; any object with .start/.tick/.done/.log
    max_pages_per_group: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create one multi-page PDF per group from page images.
    """

    # ---------- local helpers (no external deps) ----------
    def _sort_members(members: List[str]) -> List[str]:
        # stable sort by numeric suffix if present; fallback to lexicographic
        def key(m: str):
            m = m.strip()
            digits = re.findall(r"(\d+)", m)
            return (int(digits[-1]) if digits else 10**9, m)
        return sorted(members, key=key)

    def _sanitize_filename(s: str) -> str:
        s = s.strip().replace(" ", "_")
        return re.sub(r"[^A-Za-z0-9._-]+", "", s) or "group"

    def _unique_path(base: Path) -> Path:
        if not base.exists():
            return base
        stem, suf = base.stem, base.suffix
        i = 2
        while True:
            cand = base.with_name(f"{stem}-{i}{suf}")
            if not cand.exists():
                return cand
            i += 1

    # ------------------------------------------------------

    image_dir = Path(image_folder)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = groups_dict.get("groups", {}) or {}
    manifest: Dict[str, Any] = {"output_dir": str(out_dir), "groups": {}}

    if reporter and hasattr(reporter, "start"):
        reporter.start("pdfs", total=len(groups), label="Writing group PDFs")

    for idx, (gid, g) in enumerate(groups.items(), start=1):
        # dedupe + deterministic ordering
        members = _sort_members(list(dict.fromkeys(g.get("members", []) or [])))
        if max_pages_per_group is not None:
            members = members[:max_pages_per_group]

        rep = g.get("rep", gid) or gid
        safe_name = _sanitize_filename(str(rep))
        pdf_path = _unique_path(out_dir / f"{safe_name}.pdf")

        pages: List[Image.Image] = []
        missing: List[str] = []

        # Collect images
        for base in members:
            p = _find_image_for_basename(image_dir, base)  # you already defined this helper
            if p is None:
                missing.append(base)
                continue
            try:
                im = _load_rgb(p)  # you already defined this helper
                pages.append(im)
            except Exception:
                missing.append(base)

        if not pages:
            manifest["groups"][gid] = {
                "rep": rep,
                "pdf": None,
                "written_pages": 0,
                "missing_members": missing
            }
            if reporter and hasattr(reporter, "tick"):
                reporter.tick("pdfs", message=f"{gid}: 0 pages (missing images)")
            continue

        # Write multi-page PDF
        try:
            cover, rest = pages[0], pages[1:]
            cover.save(pdf_path, save_all=True, append_images=rest, format="PDF")
            manifest["groups"][gid] = {
                "rep": rep,
                "pdf": str(pdf_path),
                "written_pages": len(pages),
                "missing_members": missing
            }
            if reporter and hasattr(reporter, "tick"):
                reporter.tick("pdfs", message=f"{gid}: wrote {len(pages)} page(s) → {pdf_path.name}")
        except Exception as e:
            manifest["groups"][gid] = {
                "rep": rep,
                "pdf": None,
                "written_pages": 0,
                "missing_members": members,
                "error": str(e),
            }
            if reporter and hasattr(reporter, "tick"):
                reporter.tick("pdfs", message=f"{gid}: ❌ failed ({e})")
        finally:
            # free memory
            for im in pages:
                try:
                    im.close()
                except Exception:
                    pass

    if reporter and hasattr(reporter, "done"):
        reporter.done("pdfs", success=True, message=f"PDFs saved in {out_dir}")

    return manifest



def analyze(files, reporter: Reporter = NULL_REPORTER):
    summaries = []
    results_by_file = {}
    THRESHOLD = 0.95  # keep in one place

    for file_id, v in files.items():
        try:
            pdf_path   = Path(v["paths"]["original"]).resolve()
            thumbs_dir = Path(v["paths"]["thumbnails"]).resolve()
            pages_dir  = Path(v["paths"]["pages"]).resolve()
            ocr_dir    = Path(v["paths"]["ocr_text"]).resolve()
            db_dir     = ocr_dir  # vector DB alongside OCR (your current design)

            for d in (thumbs_dir, pages_dir, ocr_dir):
                d.mkdir(parents=True, exist_ok=True)

            # -------- File header --------
            with fitz.open(str(pdf_path)) as doc:
                page_count = doc.page_count
            file_label = f"{v.get('name', file_id)} ({page_count} pages)"
            reporter.start("file", label=f"Processing {file_label}")
            reporter.log("file", f"➡️ {file_label}")

            # -------- 1) Render pages + thumbnails --------
            reporter.start("render", total=page_count, label="Rendering pages")
            written_pages, written_thumbs = render_pages_if_needed(
                pdf_path=str(pdf_path),
                thumbs_dir=str(thumbs_dir),
                pages_dir=str(pages_dir),
                dpi=300,
                thumb_size=(320, 320),
                prefix="page_",
                zero_pad=4,
                # reporter=reporter,  # <- enable when helper supports it
            )
            reporter.done("render", message=f"Rendered {len(written_pages)} new page(s)")

            summaries.append({
                "file_id": file_id,
                "pdf_path": str(pdf_path),
                "pages_dir": str(pages_dir),
                "thumbs_dir": str(thumbs_dir),
                "ocr_dir": str(ocr_dir),
                "page_count": page_count,
                "new_pages": written_pages,
                "new_thumbs": written_thumbs,
            })

            # -------- 2) OCR (idempotent) --------
            # estimate total PNGs for progress label
            pngs_total = len([p for p in pages_dir.glob("*.png")])
            reporter.start("ocr", total=pngs_total, label="Running OCR")
            # ocr_pages_if_needed(pages_dir, ocr_dir, reporter=reporter)  # <- when helper supports it
            ocr_pages_if_needed(pages_dir, ocr_dir)
            reporter.done("ocr", message="OCR complete")

            # -------- 3) Vector DB (idempotent) --------
            # Count OCR texts to show in status
            txt_count = len(list(ocr_dir.glob("*.txt")))
            reporter.start("vectorize", total=txt_count or None, label="Embedding & indexing")
            build_info = build_vector_db(ocr_dir=str(ocr_dir), db_dir=str(db_dir))
            if isinstance(build_info, dict) and build_info.get("error"):
                reporter.done("vectorize", success=False, message=build_info["error"])
                # No OCR texts or other issue—capture and continue to next file
                results_by_file[file_id] = {
                    "groups_json": None,
                    "groups": {"groups": {}},
                    "group_pdfs_dir": None,
                    "error": build_info["error"],
                }
                reporter.done("file", success=False, message=f"Skipped {file_label}")
                continue
            reporter.done("vectorize", message=f"Indexed {build_info.get('count', txt_count)} page(s)")

            # -------- 4) Load DB --------
            reporter.start("load_db", label="Loading vector DB")
            _db = load_vector_db(str(db_dir))
            reporter.done("load_db", message="Vector DB loaded")

            # -------- 5) Group near-duplicates --------
            reporter.start("group", label="Grouping near-duplicates")
            groups = group_near_duplicates(str(db_dir), threshold=THRESHOLD)
            groups_count = len((groups or {}).get("groups", {}))
            reporter.done("group", message=f"Found {groups_count} group(s)")

            # -------- 6) Persist groups JSON --------
            output_path = db_dir / f"{pdf_path.stem}_duplicate_groups.json"
            output_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")

            # -------- 7) Create group PDFs --------
            group_pdfs_dir = pages_dir / "group_pdfs"
            group_pdfs_dir.mkdir(parents=True, exist_ok=True)
            reporter.start("pdfs", total=groups_count or None, label="Writing group PDFs")
            # pass reporter when your create_group_pdfs accepts it:
            # manifest = create_group_pdfs(groups, image_folder=str(pages_dir), output_dir=str(group_pdfs_dir), reporter=reporter)
            manifest = create_group_pdfs(groups, image_folder=str(pages_dir), output_dir=str(group_pdfs_dir))
            reporter.done("pdfs", message=f"PDFs saved in {group_pdfs_dir}")

            # -------- Collect final result --------
            results_by_file[file_id] = {
                "groups_json": str(output_path),
                "groups": groups,
                "groups_count": groups_count,
                "group_pdfs_dir": str(group_pdfs_dir),
                "pdf_manifest": manifest,
            }

            reporter.done("file", message=f"✅ Done: {file_label}")

        except Exception as e:
            err = str(e)
            summaries.append({
                "file_id": file_id,
                "error": err,
                "traceback": traceback.format_exc()
            })
            results_by_file[file_id] = {
                "groups_json": None,
                "groups": {"groups": {}},
                "group_pdfs_dir": None,
                "error": err,
            }
            # mark current stage/file as failed visibly
            reporter.done("file", success=False, message=f"❌ {file_label}: {err}")

    return {
        "threshold": THRESHOLD,
        "summaries": summaries,
        "results": results_by_file
    }





def get_groups():
    # Return cached groups if available; else empty dict
    return jsonify({"groups": groups_cache}), 200

def clear():
    global uploaded_files, groups_cache
    # remove uploaded PDFs and per-file folders
    for meta in list(uploaded_files.values()):
        try:
            if os.path.exists(meta["path"]):
                os.remove(meta["path"])
        except Exception:
            pass
        # remove per-file dirs
        base, pages, ocr, thumbs = file_dirs(meta["id"])
        for d in (pages, ocr, thumbs, base):
            try:
                if os.path.isdir(d):
                    # remove files inside
                    for fn in os.listdir(d):
                        try:
                            os.remove(os.path.join(d, fn))
                        except Exception:
                            pass
                    os.rmdir(d)
            except Exception:
                pass

    # clear index dir
    try:
        for fn in os.listdir(INDEX_DIR):
            try:
                os.remove(os.path.join(INDEX_DIR, fn))
            except Exception:
                pass
    except Exception:
        pass

    uploaded_files = {}
    groups_cache = {}
    return jsonify({"message": "Cleared all data"}), 200


def serve_uploads(filename):
    return send_from_directory(UPLOAD_ROOT, filename)


def health():
    return jsonify({
        "uploaded_files": len(uploaded_files),
        "groups_found": len(groups_cache),
    }), 200