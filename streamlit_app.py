# streamlit_app.py — Brain Tumour Analysis (UI updates + compact sidebar chat card)
# Run: streamlit run streamlit_app.py --server.port 8510

from __future__ import annotations
import io, time, html
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import streamlit as st

def _st_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import scipy.ndimage as ndi

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
(OUTPUTS_DIR / "overlays").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "gradcam").mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 192
TILE_SIZE    = 360

DEFAULT_TAU = 0.80
DEFAULT_MIN_AREA_PCT = 0.8  # brain-relative

st.set_page_config(page_title="Brain Tumour Analysis", page_icon="🧠", layout="wide")

# ---------- CSS (UI ONLY) ----------
st.markdown("""
<style>
.block-container { padding-top: 1.8rem; padding-bottom: 0.4rem; }
.block-container h1 { font-size: 2.18rem; line-height: 1.28; margin: 0.6rem 0 0.35rem 0; }

/* Cards */
.card { border: 1px solid #000; border-radius: 10px; padding: 8px 10px; margin-bottom: 10px; background: #fff; margin-top: .7rem!important; }
.card h5 { margin: 0 0 6px 0; font-weight: 700; font-size: .95rem; }
.card .cap { color:#4b4f56; font-size:.80rem; }
.card.tight { padding: 4px 8px; margin-top: .55rem!important; }
.card.tight h5 { margin-bottom: 3px; font-size: .93rem; }

/* Status rows */
.status-row { display:flex; gap:8px; margin:2px 0; }
.status-row .key { width:86px; font-weight:700; }
.status-row .val { flex:1; }

/* Sidebar frame */
section[data-testid="stSidebar"]{ border-right:2px solid #000!important; padding-left:12px!important; min-width:360px; max-width:360px; }

/* Slider chips + divider */
.sb-top { position:relative; padding:6px 2px 0; }
.sb-chip{ display:inline-block; border:1px solid #000; border-radius:8px; padding:6px 10px; font-weight:700; background:#f5f7fb; margin:0 0 6px 2px; font-size:.92rem; }
.sb-top .sb-vline{ position:absolute; top:12px; bottom:10px; left:50%; width:2px; background:#000; pointer-events:none; }
section[data-testid="stSidebar"] hr.sb-hr{ border:0!important; height:0!important; margin:12px 0 14px!important; position:relative!important; }
section[data-testid="stSidebar"] hr.sb-hr::before{ content:""; position:absolute; top:0; left:calc(-1rem - 12px); right:calc(-1rem - 12px); border-top:2px solid #000; }

/* Chatbox (self-contained bordered card) */

/* Remove any automatic Streamlit container borders in the sidebar */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]{
  border:0 !important;
  box-shadow:none !important;
  background:transparent !important;
  padding:0 !important;
}

/* Hide accidental empty buttons in the sidebar */
section[data-testid="stSidebar"] .stButton > button:empty{
  display:none !important;
}


/* Round-pill buttons (chat only) */
.sb-chat .stButton > button[kind="secondary"]{
  background:#fff!important;
  color:#184780!important;
  border:1px solid #000!important;
  border-radius:999px!important;
  padding:4px 12px!important;
  font-weight:700!important;
  font-size:.85rem!important;
  box-shadow:none!important;
}
.sb-chat .stButton > button[kind="secondary"]:hover{
  background:#f5f7fb!important;
}
/* Chat scroll area (little smaller) */
.sb-chat .drawer-body{ height:24vh; overflow:auto; margin:8px 0; border:1px solid #e7e7e7; border-radius:8px; padding:8px; background:#f9fbff; }

/* chat bubbles */
.chat-wrap{ font-size:.95rem; }
.chat-row{ display:flex; margin:6px 0; }
.chat-row.bot{ justify-content:flex-start; }
.chat-row.user{ justify-content:flex-end; }
.chat-bubble{ max-width:92%; padding:10px 14px; border-radius:16px; border:1px solid #cfd9e6; white-space:pre-wrap; word-wrap: break-word; }
.chat-bubble.bot{ background:#e8f5ff; color:#184780; border-color:#bcd2e6; }
.chat-bubble.user{ background:#fff; color:#333; border-color:#d6d6d6; }
.chat-typing{ opacity:.85; }

/* Quick replies */
.sb-chat #qr-zone button{ width:100%!important; margin-bottom:6px!important; text-align:left!important; }

/* Chat input */
.sb-chat .drawer-input form{ display:flex; gap:6px; }
.sb-chat .drawer-input input{ flex:1; }

/* PDF button */
div#pdf-btn button{ background:#E53935!important; color:#fff!important; border:1px solid #000!important; padding:.28rem .6rem!important; border-radius:8px!important; font-size:.88rem!important; }

/* Misc spacing */
[data-testid="stVerticalBlock"]{ gap:.5rem; }
button[kind="primary"]{ margin-bottom:1rem!important; }
img{ border-radius:8px; }
.imgcap{ color:#444; font-weight:700; font-size:.95rem; margin:6px 0; }
.imgcap.center{ text-align:center; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("Brain Tumour Analysis")
st.caption("Research prototype")
with st.expander("About", expanded=False):
    st.markdown(
        "- **Presence**: binary ResNet50 (`classifier_binary_resnet50.pt`)\n"
        "- **Type**: multiclass ResNet50 (`classifier_resnet50.pt`)\n"
        "- **Segmentation**: U-Net (`unet_deeper.pt`) with τ cutoff + min-area gate\n"
        "- **Explainability**: Grad-CAM on classifier\n"
        "- **Staging**: **% of brain area** — Early < 3% | Medium 3–6% | Advanced > 6%\n"
        "- **Prognostics**: demo rule-based estimate from type + stage"
    )

# ---------- Models / helpers (unchanged logic) ----------
def build_resnet50_head(num_classes: int):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.eval()
    return m

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, depth=5):
        super().__init__()
        chans = [base*(2**i) for i in range(depth)]
        self.enc = nn.ModuleList([DoubleConv(in_ch, chans[0])])
        self.pool = nn.ModuleList()
        for i in range(depth-1):
            self.pool.append(nn.MaxPool2d(2))
            self.enc.append(DoubleConv(chans[i], chans[i+1]))
        self.up  = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up.append(nn.ConvTranspose2d(chans[i+1], chans[i], 2, 2))
            self.dec.append(DoubleConv(chans[i]*2, chans[i]))
        self.head = nn.Conv2d(chans[0], out_ch, 1)
    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.pool):
                skips.append(x); x = self.pool[i](x)
        for i in range(len(self.up)):
            x = self.up[i](x)
            skip = skips[-(i+1)]
            if x.shape[-2:] != skip.shape[-2:]:
                dy = skip.shape[-2] - x.shape[-2]
                dx = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            x = self.dec[i](torch.cat([skip, x], dim=1))
        return self.head(x)

@st.cache_resource
def load_binary_model():
    ck = MODELS_DIR / "classifier_binary_resnet50.pt"
    if not ck.exists():
        return None, None
    state = torch.load(ck, map_location="cpu")
    classes = state.get("classes", ["no_tumor","tumor_present"])
    m = build_resnet50_head(2)
    m.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return m.to(DEVICE).eval(), classes

@st.cache_resource
def load_multiclass_model():
    ck = MODELS_DIR / "classifier_resnet50.pt"
    if not ck.exists():
        return None, None
    state = torch.load(ck, map_location="cpu")
    classes = state.get("classes", ["glioma","meningioma","no_tumor","pituitary"])
    m = build_resnet50_head(len(classes))
    m.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return m.to(DEVICE).eval(), classes

@st.cache_resource
def load_unet():
    ck = MODELS_DIR / "unet_deeper.pt"
    if not ck.exists():
        return None
    m = UNet(3,1,base=32,depth=5)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    return m.to(DEVICE).eval()

BIN_MODEL, BIN_CLASSES = load_binary_model()
MC_MODEL,  MC_CLASSES  = load_multiclass_model()
UNET_MODEL             = load_unet()

def tfm_img224():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def tfm_imgN(n=IMG_SIZE_SEG):
    return transforms.Compose([transforms.Resize((n,n)), transforms.ToTensor()])

def fit_square(pil_img, size=TILE_SIZE):
    return ImageOps.fit(pil_img, (size, size), method=Image.LANCZOS)

def postprocess_lcc(bin_np):
    lab, n = ndi.label(bin_np.astype(np.uint8))
    if n == 0:
        return bin_np
    sizes = ndi.sum(bin_np, lab, index=range(1, n+1))
    keep = int(np.argmax(sizes) + 1)
    out = (lab == keep).astype(np.uint8)
    out = ndi.binary_opening(out, structure=np.ones((3,3))).astype(np.uint8)
    return out

def estimate_brain_mask(pil_img, out_size=IMG_SIZE_SEG):
    g = pil_img.convert("L").resize((out_size, out_size))
    g = np.asarray(g, dtype=np.float32) / 255.0
    thr = max(0.10, float(np.percentile(g[g > 0], 30))) if np.any(g > 0) else 0.10
    m = (g > thr).astype(np.uint8)
    m = ndi.binary_opening(m, structure=np.ones((3,3))).astype(np.uint8)
    m = ndi.binary_closing(m, structure=np.ones((5,5))).astype(np.uint8)
    lab, n = ndi.label(m)
    if n > 0:
        sizes = ndi.sum(m, lab, index=range(1, n+1))
        keep = int(np.argmax(sizes) + 1)
        m = (lab == keep).astype(np.uint8)
    return m

def gradcam_overlay(model, img_pil, class_idx=None, img_size=IMG_SIZE_CLS):
    model.eval()
    target_layer = model.layer4[-1].conv3
    x = tfm_img224()(img_pil).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    acts, grads = [], []
    def fwd_hook(_, __, out): acts.append(out)
    def bwd_hook(_, gin, gout): grads.append(gout[0])
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    with torch.enable_grad():
        out = model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        score = out[0, class_idx]
        model.zero_grad(set_to_none=True)
        score.backward()
        A = acts[0]; G = grads[0]
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1)
        cam = F.relu(cam)[0].detach().cpu().numpy()
    h1.remove(); h2.remove()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    import matplotlib.cm as cm
    vis = img_pil.resize((img_size, img_size)).convert("RGB")
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.blend(vis, Image.fromarray(heat).resize(vis.size), alpha=0.45)
    return overlay

def estimate_survival(tumor_type: str, stage: str):
    TABLE = {"glioma":{"Early":0.80,"Medium":0.55,"Advanced":0.25},
             "meningioma":{"Early":0.95,"Medium":0.85,"Advanced":0.65},
             "pituitary":{"Early":0.97,"Medium":0.92,"Advanced":0.85},
             "no_tumor":{"Early":0.99,"Medium":0.99,"Advanced":0.99}}
    if tumor_type not in TABLE or not stage:
        return None
    return 100.0 * float(TABLE[tumor_type][stage])

@torch.inference_mode()
def predict_binary(pil_img):
    if BIN_MODEL is None:
        return None, None
    x = tfm_img224()(pil_img).unsqueeze(0).to(DEVICE)
    logits = BIN_MODEL(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx = int(np.argmax(probs))
    return BIN_CLASSES[idx], float(probs[idx])

@torch.inference_mode()
def predict_multiclass(pil_img):
    if MC_MODEL is None:
        return None, None, None
    x = tfm_img224()(pil_img).unsqueeze(0).to(DEVICE)
    logits = MC_MODEL(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx = int(np.argmax(probs))
    return MC_CLASSES[idx], float(probs[idx]), {cls: float(p) for cls,p in zip(MC_CLASSES, probs)}

@torch.inference_mode()
def run_unet(pil_img, tau):
    if UNET_MODEL is None:
        return None, None, None, None, None
    x = tfm_imgN()(pil_img).unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(UNET_MODEL(x))[0,0].detach().cpu().numpy()
    pred = (prob > float(tau)).astype(np.uint8)
    pred = postprocess_lcc(pred)
    brain = estimate_brain_mask(pil_img, out_size=IMG_SIZE_SEG)
    tumor_px = float(pred.sum()); img_px = float(pred.size)
    brain_px = float(brain.sum()) if brain is not None else 0.0
    area_img   = 100.0 * tumor_px / max(img_px,   1.0)
    area_brain = 100.0 * tumor_px / max(brain_px, 1.0)
    ys, xs = np.where(pred > 0.5)
    if len(xs) > 0:
        cx, cy = float(xs.mean()), float(ys.mean())
        cx_n, cy_n = cx / pred.shape[1], cy / pred.shape[0]
    else:
        cx = cy = cx_n = cy_n = float('nan')
    base = pil_img.resize((IMG_SIZE_SEG, IMG_SIZE_SEG)).convert('RGB')
    ov = base.copy(); draw = ImageDraw.Draw(ov, "RGBA")
    for X, Y in zip(xs, ys):
        draw.point((X,Y), fill=(255,0,0,100))
    return ov, float(area_img), float(area_brain), {"x_px":cx,"y_px":cy,"x":cx_n,"y":cy_n}, pred

# ---------- Sidebar inputs ----------
st.sidebar.markdown('<div class="sb-top">', unsafe_allow_html=True)
colL, colR = st.sidebar.columns(2)
with colL:
    st.markdown('<span class="sb-chip">Seg. cutoff (τ)</span>', unsafe_allow_html=True)
    tau = st.slider("Seg. cutoff (τ)", 0.10, 0.95, DEFAULT_TAU, 0.01, label_visibility="collapsed")
with colR:
    st.markdown('<span class="sb-chip">Min area % (brain)</span>', unsafe_allow_html=True)
    min_area_pct = st.slider("Min area % (brain)", 0.0, 5.0, DEFAULT_MIN_AREA_PCT, 0.1, label_visibility="collapsed")
st.sidebar.markdown('<div class="sb-vline"></div></div>', unsafe_allow_html=True)
st.sidebar.markdown('<hr class="sb-hr" />', unsafe_allow_html=True)

# ---------- Small chat state ----------
ss = st.session_state
ss.setdefault("chat_msgs", [])
ss.setdefault("mode", None)             # None | "explain" | "pdf" | "done"
ss.setdefault("presence", None)         # "Yes" | "No" | None
ss.setdefault("rtype", None)
ss.setdefault("rconf", None)
ss.setdefault("area_val", None)
ss.setdefault("stage", None)
ss.setdefault("await_explain", False)
ss.setdefault("await_conf", False)
ss.setdefault("await_area", False)
ss.setdefault("await_stage", False)
ss.setdefault("await_visuals", False)
ss.setdefault("await_end", False)
ss.setdefault("pdf_step", 0)
ss.setdefault("pending_queue", [])      # queue of bot messages

def _bot_queue(text: str):
    ss["pending_queue"].append(text)

def _summary_text():
    if ss["presence"] == "Yes" and ss["rtype"]:
        parts = [f"Summary: Presence — Tumour", f"Type — {ss['rtype']}"]
        if ss["rconf"] is not None:
            parts[-1] += f" (conf {ss['rconf']:.2f})"
        if ss["area_val"] is not None:
            parts.append(f"Area — {ss['area_val']:.2f}% of brain")
        if ss["stage"]:
            parts[-1] = parts[-1] + f" • Stage — {ss['stage']}"
        return " | ".join(parts)
    if ss["presence"] == "No":
        return "Summary: Presence — No tumour" + (f" (conf {ss['rconf']:.2f})" if ss["rconf"] is not None else "") + "."
    return "Summary: Presence — Unknown."

def _reset_to_menu():
    ss["mode"]=None; ss["presence"]=None; ss["rtype"]=None; ss["rconf"]=None
    ss["area_val"]=None; ss["stage"]=None
    ss["await_explain"]=False; ss["await_conf"]=False; ss["await_area"]=False
    ss["await_stage"]=False; ss["await_visuals"]=False; ss["await_end"]=False
    ss["pdf_step"]=0; ss["pending_queue"].clear()
    if len(ss["chat_msgs"])==0:
        ss["chat_msgs"].append({"role":"bot","text":"Hi! I can explain your report in simple English.\nI’m an educational assistant, not a clinician."})
    _bot_queue("Please choose one option:\n1) Explain my report\n2) Help with downloading the report")

def _close_done(text="Okay — bye! If you have any questions, ask me any time. 👋"):
    _bot_queue(text)
    ss["mode"]="done"; ss["await_explain"]=ss["await_conf"]=ss["await_area"]=False
    ss["await_stage"]=ss["await_visuals"]=ss["await_end"]=False; ss["pdf_step"]=0

def _start_over():
    ss["chat_msgs"].clear(); _reset_to_menu()

def _is_float_str(s:str)->bool:
    try:
        float(str(s).strip()); return True
    except:
        return False

def _handle_user(msg: str):
    qraw = msg.strip(); q = qraw.lower()
    if q=="menu":
        _reset_to_menu(); return
    if q in {"stop","finish"} and ss["mode"]!="pdf":
        _close_done(); return

    if ss["mode"] is None:
        if q in {"1","explain","1) explain my report"}:
            ss["mode"]="explain"
            _bot_queue("What does your report say about presence?\n1) Tumour present\n2) No tumour\n\nType 1 or 2 (or 'menu' to go back)."); return
        if q in {"2","pdf help","download","2) help with downloading the report"}:
            ss["mode"]="pdf"; ss["pdf_step"]=1
            _bot_queue("Do these steps to download your PDF:\n1. Scroll to the bottom of the results panel on the right.\n2. Find the red ⬇️ 'Download PDF' button.\n3. Click it once — your browser will save the file (Downloads bar or top-right downloads icon).\n\nDid you find and download it?"); return
        _reset_to_menu(); return

    if ss["mode"]=="pdf":
        if ss["pdf_step"]==1:
            if q in {"yes","y","downloaded","done","pdf_yes"}:
                ss["pdf_step"]=2; _bot_queue("Great! Would you like me to explain your report now?\nReply 'explain' to continue, or 'finish' to end.")
            else:
                _bot_queue("Tap **Yes / Downloaded** when it’s saved, then we can continue.")
            return
        if ss["pdf_step"]==2:
            if q=="explain":
                _close_done("To explain your report, tap **Go to menu (start over)** below, then choose **Explain my report**."); return
            if q in {"finish","stop"}:
                _close_done(); return
            _bot_queue("Please type 'explain' to continue, or 'finish' to end."); return

    if ss["mode"]=="explain" and ss["presence"] is None:
        if q in {"1","yes","tumour present","tumor present","tumour","tumor"}:
            ss["presence"]="Yes"; _bot_queue("Which type does it mention?\n• glioma • meningioma • pituitary"); return
        if q in {"2","no","no tumour","no tumor"}:
            ss["presence"]="No"; ss["await_conf"]=True
            _bot_queue("What confidence / probability does your report show?\n(Enter a number like 0.97.)"); return
        _bot_queue("Please choose:\n1) Tumour present\n2) No tumour"); return

    if ss["mode"]=="explain" and ss["presence"]=="Yes" and ss["rtype"] is None:
        if q in {"glioma","meningioma","pituitary","1","2","3"}:
            ss["rtype"] = "glioma" if q in {"1","glioma"} else "meningioma" if q in {"2","meningioma"} else "pituitary"
            ss["await_explain"]=True; _bot_queue(f"Would you like a quick explanation of {ss['rtype']}?\n1) Yes, explain\n2) Skip"); return
        _bot_queue("Please type one of: glioma / meningioma / pituitary"); return

    if ss["await_explain"]:
        yes_set={"1","1) yes, explain","yes","y","ok","okay","sure","explain","yes, explain"}
        skip_set={"2","2) skip","skip","no","n"}
        if q in yes_set:
            EXPLAIN={
                "meningioma":"Meningioma (plain English)\n• Grows from the coverings of the brain (meninges).\n• Many are non-cancerous and often slow-growing; next steps depend on size and location.",
                "glioma":"Glioma (plain English)\n• Starts from the brain’s support cells (glia).\n• Some grow slowly and others faster; plans depend on how active it looks and where it sits.",
                "pituitary":"Pituitary tumour (plain English)\n• Arises in the small gland that controls hormones.\n• It can affect vision or hormones; treatment and follow-up are clinician-led.",
            }
            _bot_queue(EXPLAIN.get(ss["rtype"], ""))
            ss["await_explain"]=False; ss["await_conf"]=True
            _bot_queue("What confidence / probability does your report show?\n(Enter a number like 0.88.)"); return
        if q in skip_set:
            ss["await_explain"]=False; ss["await_conf"]=True
            _bot_queue("What confidence / probability does your report show?\n(Enter a number like 0.88.)"); return
        _bot_queue('Please reply "1" for Yes, explain or "2" to Skip.'); return

    if ss["await_conf"]:
        if _is_float_str(q):
            ss["rconf"]=float(q); ss["await_conf"]=False
            if ss["presence"]=="Yes":
                ss["await_area"]=True; _bot_queue("What % of the brain area does the report show for the tumour?\n(Enter a number like 1.84.)")
            else:
                _bot_queue(_summary_text()); ss["await_end"]=True
                _bot_queue("Anything else?\n1) Help with downloading the report\n2) Finish")
        else:
            _bot_queue("Please enter a number like 0.88")
        return

    if ss["await_area"]:
        if _is_float_str(q):
            ss["area_val"]=float(q); ss["await_area"]=False; ss["await_stage"]=True
            _bot_queue("What stage does the report state for the tumour?\n(Early / Medium / Advanced)")
        else:
            _bot_queue("Please enter a number like 1.84")
        return

    if ss["await_stage"]:
        m={"early":"Early","medium":"Medium","advanced":"Advanced"}
        if q in m:
            ss["stage"]=m[q]; ss["await_stage"]=False
            _bot_queue(_summary_text())
            ss["await_visuals"]=True
            _bot_queue("Would you like me to explain the Visuals and Grad-CAM?\n1) Yes\n2) No")
        else:
            _bot_queue("Please type one of: Early / Medium / Advanced")
        return

    if ss["await_visuals"]:
        if q in {"1","yes","y"}:
            _bot_queue("Visuals & Grad-CAM (short)\n\n• Original — the raw MRI image.\n• Overlay — red shading shows where the system believes tumour tissue is located and its approximate area.\n• Mask — the binary shape of the predicted tumour region (shows its outline/shape).\n• Grad-CAM — highlights where the classifier “looked”; warm colours should roughly overlap the red overlay area.")
            ss["await_visuals"]=False; ss["await_end"]=True
            _bot_queue("Anything else?\n1) Help with downloading the report\n2) Finish"); return
        if q in {"2","no","n"}:
            ss["await_visuals"]=False; ss["await_end"]=True
            _bot_queue("Got it.\nAnything else?\n1) Help with downloading the report\n2) Finish"); return
        _bot_queue('Please reply "1" for Yes or "2" for No.'); return

    if ss["await_end"]:
        if q in {"1","help","help with downloading the report","pdf help","download"}:
            _close_done("Tap **Go to menu (start over)** below, then choose **PDF help**.\nIt will show the steps to find the red ⬇️ Download PDF button."); return
        if q in {"2","finish","done","stop"}:
            _close_done(); return
        _bot_queue("Please choose:\n1) Help with downloading the report\n2) Finish"); return

    _bot_queue("Type 'menu' to start again, or continue with the shown options.")

# ---------- Render sidebar chat (COMPACT CARD) ----------
with st.sidebar:
    # This block for the word "Assistant" and its container
    st.markdown('<div style="text-align:center; margin-bottom: 10px;"><div style="border:1px solid #000; border-radius:8px; padding:4px 8px; background:#fff; display:inline-block;">Assistant</div></div>', unsafe_allow_html=True)

    # This is the main chat container that holds everything else below
    st.markdown('<div class="sb-chat">', unsafe_allow_html=True)

    # The rest of the chat interface starts here
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        qs1 = st.button("Explain", key="qs_explain_sb", type="secondary")
    with col_b:
        qs2 = st.button("PDF help", key="qs_pdf_sb", type="secondary")

    if qs1:
        ss.chat_msgs.append({"role": "user", "text": "1) Explain my report"})
        _handle_user("explain")
        _st_rerun()

    if qs2:
        ss.chat_msgs.append({"role": "user", "text": "2) Help with downloading the report"})
        _handle_user("pdf help")
        _st_rerun()
    
    # ... The rest of your chat code should follow here, indented inside the with st.sidebar block ...
    # messages
    body_html = ['<div class="drawer-body"><div class="chat-wrap">']
    if len(ss["chat_msgs"]) == 0:
        _reset_to_menu()

    for m in ss["chat_msgs"]:
        txt = html.escape(m["text"])
        role = m["role"]
        if role == "bot":
            body_html.append(
                f'<div class="chat-row bot"><div class="chat-bubble bot">{txt}</div></div>'
            )
        else:
            body_html.append(
                f'<div class="chat-row user"><div class="chat-bubble user">{txt}</div></div>'
            )
    body_html.append("</div></div>")
    st.markdown("".join(body_html), unsafe_allow_html=True)

    # typing… (drain one queued message per rerun)
    if ss.get("pending_queue"):
        next_msg = ss["pending_queue"].pop(0)
        ph = st.empty()
        for dots in ["Typing", "Typing.", "Typing..", "Typing..."]:
            ph.markdown(
                '<div class="drawer-body"><div class="chat-wrap">'
                '<div class="chat-row bot"><div class="chat-bubble bot chat-typing">'
                + html.escape(dots) +
                "</div></div></div></div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.22)
        ph.empty()
        ss["chat_msgs"].append({"role": "bot", "text": next_msg})
        _st_rerun()

    # quick replies
    def _quick_replies_for_state():
        if ss["mode"] == "done":
            return []
        if ss["mode"] is None:
            return [("1) Explain my report", "explain"),
                    ("2) Help with downloading the report", "pdf help")]
        if ss["mode"] == "pdf":
            return [("Yes / Downloaded", "pdf_yes")] if ss["pdf_step"] == 1 else []
        if ss["mode"] == "explain":
            if ss["presence"] is None:
                return [("1) Tumour present", "1"), ("2) No tumour", "2")]
            if ss["presence"] == "Yes" and ss["rtype"] is None:
                return [("glioma", "glioma"), ("meningioma", "meningioma"), ("pituitary", "pituitary")]
            if ss["await_explain"]:
                return [("1) Yes, explain", "1"), ("2) Skip", "2")]
            if ss["await_stage"]:
                return [("Early", "early"), ("Medium", "medium"), ("Advanced", "advanced")]
            if ss["await_visuals"]:
                return [("1) Yes", "1"), ("2) No", "2")]
            if ss["await_end"]:
                return [("1) Help with downloading the report", "1"), ("2) Finish", "2")]
        return []

    qrs = _quick_replies_for_state()
    if qrs:
        st.markdown('<div id="qr-zone">', unsafe_allow_html=True)
        for i, (label, payload) in enumerate(qrs):
            if st.button(label, key=f"qr_sb_{i}_{abs(hash(label)) % 1000000}"):
                ss["chat_msgs"].append({"role": "user", "text": label})
                _handle_user(payload)
                _st_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # input
    if ss["mode"] == "done":
        st.markdown('<div class="drawer-input">', unsafe_allow_html=True)
        if st.button("Go to menu (start over)", key="btn_restart_chat"):
            _start_over()
            _st_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        with st.form("chat_drawer_form_sb", clear_on_submit=True):
            st.markdown('<div class="drawer-input">', unsafe_allow_html=True)
            utext = st.text_input(
                "Type here (e.g., 1 / 2 / meningioma / 0.88 / 1.84 / early / pdf help / menu / finish)",
                key="chat_input_drawer_sb",
            )
            sent = st.form_submit_button("Send")
            st.markdown("</div>", unsafe_allow_html=True)

        if sent and utext.strip():
            ss["chat_msgs"].append({"role": "user", "text": utext})
            _handle_user(utext)
            _st_rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end .sb-chat





# ---------- Uploader ----------
c1, c2, c3 = st.columns([1,2,1])
with c2:
    up = st.file_uploader("Upload MRI (PNG/JPG)", type=["png","jpg","jpeg"])
    run = st.button("Run Analysis", type="primary", use_container_width=True)
if run and up is None:
    st.warning("Please upload an image first.")

# ---------- Pipeline (unchanged) ----------
if up and run:
    img = Image.open(io.BytesIO(up.read())).convert("RGB")

    if BIN_MODEL is None:
        st.error("Binary model not found."); st.stop()
    if MC_MODEL  is None:
        st.error("Multiclass model not found."); st.stop()
    if UNET_MODEL is None:
        st.error("UNet model not found."); st.stop()

    bin_label, bin_conf = predict_binary(img)
    if bin_label != "tumor_present":
        st.markdown(
            '<div class="card"><h5>Status</h5>'
            '<div class="status-row"><span class="key">Presence</span><span class="val"> - No tumour</span></div>'
            '<div class="status-row"><span class="key">Type</span><span class="val"> - —</span></div>'
            '<div class="status-row"><span class="key">Stage</span><span class="val"> - —</span></div>'
            '<div class="status-row"><span class="key">Survival</span><span class="val"> - —</span></div>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="card"><h5>Original</h5>', unsafe_allow_html=True)
        VIS_TILE_VISUALS = int(TILE_SIZE * 0.85)
        st.image(fit_square(img, size=VIS_TILE_VISUALS))
        st.markdown("<div class='imgcap'>Original (no tumour)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        meta = {"Presence":"No tumour","Binary conf": f"{bin_conf:.3f}" if bin_conf is not None else "n/a"}
        pdf_path = OUTPUTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_no.pdf"
        ok=False
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            c.setFont("Helvetica-Bold", 16); c.drawString(40, A4[1]-60, "Brain Tumour — Case Report")
            c.setFont("Helvetica", 9); c.drawString(40, A4[1]-78, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
            c.setFont("Helvetica", 10); y = A4[1]-110
            for k,v in meta.items(): c.drawString(40, y, f"{k}: {v}"); y -= 12
            buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG"); buf.seek(0)
            c.drawImage(ImageReader(buf), 40, 120, width=220, height=220, preserveAspectRatio=True)
            c.showPage(); c.save(); ok=True
        except Exception as e:
            print("PDF generation (no tumour) failed:", e)
        if ok and pdf_path.exists():
            with open(pdf_path, "rb") as fh:
                st.markdown('<div id="pdf-btn">', unsafe_allow_html=True)
                st.download_button("⬇️ Download PDF", fh, file_name=pdf_path.name, mime="application/pdf")
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h5>Staging</h5><div class="cap">Early &lt; 3% | Medium 3–6% | Advanced &gt; 6% (of brain)</div></div>', unsafe_allow_html=True)
        st.stop()

    mc_label, mc_conf, probs_dict = predict_multiclass(img)
    gc = gradcam_overlay(MC_MODEL, img)
    ov, area_img, area_brain, centroid, pred_mask = run_unet(img, tau=float(tau))
    if area_brain < float(min_area_pct):
        stage=None
    else:
        stage = "Early" if area_brain < 3 else "Medium" if area_brain < 6 else "Advanced"
    surv = estimate_survival(mc_label, stage) if stage else None

    st.markdown(
        '<div class="card"><h5>Status</h5>'
        f'<div class="status-row"><span class="key">Presence</span><span class="val"> - Tumour</span></div>'
        f'<div class="status-row"><span class="key">Type</span><span class="val"> - {mc_label} ({mc_conf:.2f})</span></div>'
        f'<div class="status-row"><span class="key">Stage</span><span class="val"> - {("Skipped (< %.2f%% area)" % min_area_pct) if not stage else f"{stage} ({area_brain:.2f}% of brain)"}'
        '</span></div>'
        f'<div class="status-row"><span class="key">Survival</span><span class="val"> - {(f"{surv:.1f}%") if surv is not None else "Skipped"}</span></div>'
        '</div>', unsafe_allow_html=True
    )

    if isinstance(probs_dict, dict) and len(probs_dict) > 0:
        cmap = {"glioma":"#D32F2F","meningioma":"#2962FF","pituitary":"#2E7D32","no_tumor":"#616161"}
        labels = list(probs_dict.keys())
        values = [probs_dict[k] for k in labels]
        pairs = sorted(zip(values, labels), reverse=True)
        values_sorted, labels_sorted = [p[0] for p in pairs], [p[1] for p in pairs]
        colors = [cmap.get(lbl.lower(), "#777777") for lbl in labels_sorted]
        st.markdown('<div class="card tight" style="margin-top:1.2rem"><h5>Probability bar chart</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3.2, 1.25), dpi=200)
        ax.barh(labels_sorted, values_sorted, color=colors)
        ax.set_xlim(0, 1.0); ax.invert_yaxis()
        ax.set_xlabel("Probability", fontsize=6); ax.set_yticklabels(labels_sorted, fontsize=5)
        ax.tick_params(axis='x', labelsize=6)
        for v, y in zip(values_sorted, range(len(labels_sorted))):
            ax.text(min(v + 0.02, 0.98), y, f"{v:.2f}", va="center", fontsize=5)
        st.pyplot(fig, use_container_width=False); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card tight" style="margin-top:1.2rem"><h5>Visuals</h5>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    VIS_TILE_VISUALS = int(TILE_SIZE * 0.58)
    v1, v2, v3 = st.columns(3, vertical_alignment="top")
    with v1:
        st.image(fit_square(img, size=VIS_TILE_VISUALS)); st.markdown("<div class='imgcap'>Original</div>", unsafe_allow_html=True)
    with v2:
        cap = f"Overlay — area {area_brain:.2f}% of brain"
        if stage:
            cap += f" • {stage}"
        st.image(fit_square(ov, size=VIS_TILE_VISUALS)); st.markdown(f"<div class='imgcap'>{cap}</div>", unsafe_allow_html=True)
    with v3:
        mask_vis = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert("RGB")
        st.image(fit_square(mask_vis, size=VIS_TILE_VISUALS)); st.markdown("<div class='imgcap'>Mask (binary)</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:2vh'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card tight" style="margin-top:1.4rem"><h5>Grad-CAM</h5>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    VIS_TILE = int(TILE_SIZE * 0.78)
    g1, g2, g3 = st.columns(3, vertical_alignment="top")
    with g2:
        st.image(fit_square(gc, size=VIS_TILE)); st.markdown("<div class='imgcap center'>Grad-CAM</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h5>Staging</h5><div class="cap">Early &lt; 3% | Medium 3–6% | Advanced &gt; 6% (of brain)</div></div>', unsafe_allow_html=True)

    meta = {"Presence":"Tumour","Type":mc_label,"Type conf":f"{mc_conf:.3f}","Area (brain %)":f"{area_brain:.2f}",
            "Stage": stage if stage else f"Skipped (< {min_area_pct:.2f}%)",
            "Survival": f"{estimate_survival(mc_label, stage):.1f}%" if stage else "Skipped",
            "τ":f"{tau:.2f}","Min area %":f"{min_area_pct:.2f}"}
    pdf_path = OUTPUTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    ok=False
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        W,H = A4
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        c.setFont("Helvetica-Bold", 16); c.drawString(40, H-60, "Brain Tumour — Case Report")
        c.setFont("Helvetica", 9); c.drawString(40, H-78, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        c.drawString(40, H-92, "Prototype — not for clinical use")
        y = H-120; c.setFont("Helvetica", 10)
        for k,v in meta.items(): c.drawString(40, y, f"{k}: {v}"); y -= 12
        def _buf(pil): b = io.BytesIO(); pil.convert("RGB").save(b, "PNG"); b.seek(0); return b
        try: c.drawImage(ImageReader(_buf(img)), 40, 120, width=200, height=200, preserveAspectRatio=True)
        except: pass
        try: c.drawImage(ImageReader(_buf(ov)), 260, 120, width=200, height=200, preserveAspectRatio=True)
        except: pass
        try: c.drawImage(ImageReader(_buf(gc)), 480, 120, width=200, height=200, preserveAspectRatio=True)
        except: pass
        c.setFont("Helvetica-Bold", 10); c.drawString(40, 95, "Staging (relative to brain area): Early < 3% | Medium 3–6% | Advanced > 6%")
        c.showPage(); c.save(); ok=True
    except Exception as e:
        print("PDF generation failed:", e)
    if ok and pdf_path.exists():
        with open(pdf_path, "rb") as fh:
            st.markdown('<div id="pdf-btn">', unsafe_allow_html=True)
            st.download_button("⬇️ Download PDF", fh, file_name=pdf_path.name, mime="application/pdf")
            st.markdown('</div>', unsafe_allow_html=True)
