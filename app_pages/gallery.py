"""Heatmap gallery: every MVTec test image with its anomaly map, defect outline, score and verdict."""
from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from app_pages._style import inject, lede, verdict_badge

ROOT = Path(__file__).parent.parent
TILE = 384
PAGE_SIZE = 24
COLS = 4
PEAK_FILTERS = {
    "Anywhere": lambda d: d.index == d.index,
    "On the defect": lambda d: d.peak_zone == "defect",
    "Misses the defect (defects only)": lambda d: d.peak_hits_defect == False,  # noqa: E712  None = good image
    "On the background": lambda d: d.peak_zone == "background",
}
ZONE_TEXT = {"defect": "on the defect", "object": "on the product, off the defect", "background": "on the background"}

inject()
st.title("Heatmap gallery")
lede("Every test image with its anomaly map and the decision it received in cross-validation, where each image "
     "was judged by thresholds calibrated without it.")
with st.container(border=True):
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("**Green outline**  \nThe segmented product. Only heat inside it counts toward the score.")
    k2.markdown("**Magenta outline**  \nThe predicted damage: heat at or above the approve threshold, on the product.")
    k3.markdown("**White outline**  \nThe real defect, from the dataset's annotation.")
    k4.markdown("**Yellow ring**  \nThe hottest point of the map, anywhere in the photo.")

if not (ROOT / "data" / "mvtec").exists() or not list((ROOT / "reports" / "raw").glob("maps_*.npz")):
    st.warning("Needs data/mvtec (python -m cv.download_mvtec) and reports/raw/maps_*.npz from the Colab notebook.")
    st.stop()


@st.cache_resource(show_spinner="Loading anomaly maps…")
def _data():
    from claims.inspector import load_bands
    from cv.gallery import add_localization, build_table, load_maps
    from cv.heat_zones import add_zone_columns
    maps = load_maps()
    bands = load_bands(ROOT / "config" / "thresholds.json")
    return maps, add_localization(add_zone_columns(build_table(maps), maps), maps, bands), bands


@st.cache_data(max_entries=600, show_spinner=False)
def _tile(image: str, mask: str, category: str, scale: str, show_outside: bool):
    from cv.evaluate import load_mask
    from claims.segment import predicted_mask, product_region
    from cv.gallery import MAP_SIZE, load_image, render
    maps, _, bands = _data()
    full = load_image(image)
    region, _ = product_region(full)
    img = cv2.resize(full, (TILE, TILE), interpolation=cv2.INTER_AREA)
    amap = maps[image]
    return render(img, amap, load_mask(ROOT / "data" / "mvtec", mask, MAP_SIZE), bands[category], scale,
                  region=region, show_outside=show_outside, pred=predicted_mask(amap, region, bands[category].hi))


@st.cache_data(max_entries=200, show_spinner=False)
def _inset(image: str) -> tuple:
    """Zoomed, contrast-enhanced crop of the photo at the map's peak (used when the peak is on the background)."""
    import numpy as np
    from cv.gallery import load_image
    from cv.heat_zones import INSET_HALF, hotspot_crop
    maps, _, _ = _data()
    img = load_image(image)
    y, x = np.unravel_index(int(np.argmax(maps[image])), maps[image].shape)
    return hotspot_crop(img, (int(x), int(y)), half=INSET_HALF), img.shape[1] / (2 * INSET_HALF)


maps, df, bands = _data()

with st.sidebar:
    st.subheader("Filter")
    category = st.radio("Product", sorted(df.category.unique()), horizontal=True)
    d = df[df.category == category]
    types = st.multiselect("Defect type", sorted(d.defect_type.unique()), placeholder="All types")
    verdicts = st.multiselect("Out-of-fold verdict", ["APPROVE", "ESCALATE", "REJECT"], placeholder="All verdicts")
    peak = st.radio("Map peak (yellow ring)", list(PEAK_FILTERS), help=(
        "Where the anomaly map's hottest point lies. It drives the score, so a peak on the background means the "
        "verdict was set by something next to the product (on MVTec screws: lint and dust on the backdrop)."))
    sort = st.selectbox("Sort by", ["score (low → high)", "score (high → low)", "defect area (small → large)", "file"])
    scale = st.radio("Colour scale", ["fixed", "per-image"], horizontal=True,
                     help="fixed: same scale as the demo; red = at or above the approve threshold. "
                          "per-image: stretched to this image's own range, so the map's shape is always visible — "
                          "but on a good image it turns noise red.")
    show_outside = st.toggle("Show heat outside the product", value=False,
                             help="Heat outside the green outline does not count toward the score. Turn on to see "
                                  "what the model reacted to on the backdrop (for example lint and dust).")

if types:
    d = d[d.defect_type.isin(types)]
if verdicts:
    d = d[d.verdict.isin(verdicts)]
d = d[PEAK_FILTERS[peak](d)]
key, asc = {"score (low → high)": ("score", True), "score (high → low)": ("score", False),
            "defect area (small → large)": ("gt_area", True), "file": ("image", True)}[sort]
d = d.sort_values(key, ascending=asc)

band = bands[category]
defective = df[(df.category == category) & (df.label == 1)]
c1, c2, c3 = st.columns(3)
c1.metric("Images shown", f"{len(d)} / {(df.category == category).sum()}")
c2.metric("Peak on the defect", f"{int(defective.peak_hits_defect.sum())} / {len(defective)}",
          help="Pointing game: the map's hottest pixel lies on the ground-truth defect (within 6 px at 256 px).")
c3.metric("Thresholds", f"{band.lo:.1f} – {band.hi:.1f}" if band.hi > band.lo else f"{band.hi:.1f}",
          help="Reject below the first value, approve from the second, specialist review in between. "
               "A single value means there is no review band.")
if scale == "per-image":
    st.info("Per-image scale: each map is stretched to its own range. On good images this paints noise red, "
            "so compare shapes rather than colours.")

with st.expander("Predicted damage against the real defect (IoU and Dice)"):
    from cv.gallery import localization_summary
    loc = localization_summary(df[df.category == category]).reset_index(level=0, drop=True)
    loc[["median IoU", "median Dice"]] = loc[["median IoU", "median Dice"]].round(2)
    st.dataframe(loc, use_container_width=True)
    st.caption("The predicted mask is the magenta outline: heat at or above the approve threshold, inside the product. "
               "IoU and Dice measure its overlap with the white annotation (1 = identical). An empty prediction means no "
               "heat reached the approve threshold, as for every claim sent to a specialist. The threshold is the "
               "decision threshold, not one tuned for outlining defects.")

with st.expander("Peak-on-defect rate by defect type"):
    from cv.gallery import pointing_summary
    s = pointing_summary(df[df.category == category]).reset_index(level=0, drop=True)
    s["hit rate"] = s["hit rate"].map("{:.0%}".format)
    st.dataframe(s, use_container_width=True)
    approved = defective[defective.verdict == "APPROVE"]
    st.caption(f"Approved defects whose peak is elsewhere (right verdict, possibly wrong reason): "
               f"{int((approved.peak_hits_defect == False).sum())} / {len(approved)}.")  # noqa: E712

pages = max(1, -(-len(d) // PAGE_SIZE))
page = st.number_input(f"Page (of {pages})", 1, pages, 1) if pages > 1 else 1
rows = d.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
if rows.empty:
    st.info("No images match these filters.")

for start in range(0, len(rows), COLS):
    for col, r in zip(st.columns(COLS, gap="small"), rows.iloc[start:start + COLS].itertuples()):
        with col, st.container(border=True):
            st.image(_tile(r.image, r.mask, r.category, scale, show_outside), use_container_width=True)
            st.markdown(f"**{r.defect_type}/{Path(r.image).name}**")
            verdict_badge(r.verdict)
            if r.label == 1:
                where = "on the defect" if r.peak_hits_defect else ZONE_TEXT[r.peak_zone]
                st.caption(f"Score {r.score:.1f}. Defect covers {r.gt_area:.2%} of the image. Peak {where}.")
                if r.pred_px:
                    st.caption(f"Predicted vs real damage: IoU {r.iou:.2f}, Dice {r.dice:.2f}.")
                else:
                    st.caption("No predicted damage: no heat reached the approve threshold.")
            else:
                st.caption(f"Score {r.score:.1f}. No defect. Peak {ZONE_TEXT[r.peak_zone]}.")
            if r.peak_zone == "background":
                crop, zoom = _inset(r.image)
                st.image(crop, use_container_width=True,
                         caption=f"The photo at the yellow ring, zoomed {zoom:.0f}x and contrast-enhanced")
            vlm = getattr(r, "vlm_description", "")
            if isinstance(vlm, str) and vlm:
                st.caption(f"Vision-language model: {vlm}")
