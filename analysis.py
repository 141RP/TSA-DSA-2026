import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data.csv")
df["avg_score"] = df[["math_score","reading_score","writing_score"]].mean(axis=1)

# ── Palette ────────────────────────────────────────────────────────────────────
BLUSH    = "#0C0D15"
THISTLE  = "#045c5b"
LILAC    = "#55879F"
SLATE    = "#9cd4c4"
SHADOW   = "#F9F3F3"

BG       = SHADOW
CARD     = "#F9F3F3"
TEXT     = BLUSH
MUTED    = LILAC
ACCENT1  = THISTLE
ACCENT2  = LILAC
ACCENT3  = SLATE

# five categorical colors drawn from the palette + extensions
CAT5 = [THISTLE, LILAC, SLATE, "#8B7FA8", "#7A9BA8"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    SLATE,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#2E2F40",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
    "axes.titlesize":    12,
    "axes.titlecolor":   BLUSH,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  SLATE,
    "legend.fontsize":   9,
})

def label(ax, txt, x=0.0, y=1.06, size=12, color=BLUSH, weight="bold", ha="left"):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=size,
            color=color, fontweight=weight, ha=ha, va="bottom")

def subtitle(ax, txt, x=0.0, y=1.01, size=9, color=MUTED):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=size,
            color=color, ha="left", va="bottom")

def slide_header(fig, title, subtitle_txt, y_title=0.97, y_sub=0.935):
    fig.text(0.04, y_title, title, fontsize=18, fontweight="bold",
             color=BLUSH, va="top")
    fig.text(0.04, y_sub, subtitle_txt, fontsize=11, color=MUTED, va="top")

def save(fig, name):
    fig.savefig(f"{name}", dpi=160,
                bbox_inches="tight", facecolor=BG)
    print(f"  saved {name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — DEMOGRAPHIC PORTRAIT
# ══════════════════════════════════════════════════════════════════════════════
print("Slide 1 — demographic portrait")
fig = plt.figure(figsize=(16, 9), facecolor=BG)
slide_header(fig, "Who are these students...",
             "3,084 high-school students across Florida, California & Texas")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.06, right=0.97, top=0.87, bottom=0.08,
                       hspace=0.52, wspace=0.38)

# — Age distribution (area)
ax1 = fig.add_subplot(gs[0, 0])
age_counts = df["age"].value_counts().sort_index()
ax1.fill_between(age_counts.index, age_counts.values,
                 alpha=0.45, color=THISTLE)
ax1.plot(age_counts.index, age_counts.values,
         color=THISTLE, linewidth=2.2, marker="o", markersize=5)
for x, y in zip(age_counts.index, age_counts.values):
    ax1.text(x, y+6, str(y), ha="center", fontsize=8.5, color=THISTLE)
ax1.set_xticks(age_counts.index)
ax1.set_xlabel("Age")
ax1.set_ylim(0, 750)
ax1.yaxis.grid(True); ax1.set_axisbelow(True)
label(ax1, "Age distribution")
subtitle(ax1, "Roughly equal cohorts across 15–19")

# — Race breakdown (horizontal bar)
ax2 = fig.add_subplot(gs[0, 1])
race_counts = df["race"].value_counts()
bars = ax2.barh(race_counts.index, race_counts.values,
                color=CAT5, alpha=0.88, height=0.6)
for bar in bars:
    ax2.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():,}", va="center", fontsize=8.5, color=MUTED)
ax2.set_xlabel("Students")
ax2.xaxis.grid(True); ax2.set_axisbelow(True)
label(ax2, "Racial composition")
subtitle(ax2, "Five groups, evenly represented")

# — State breakdown (donut)
ax3 = fig.add_subplot(gs[0, 2])
state_counts = df["state"].value_counts()
wedges, texts, autotexts = ax3.pie(
    state_counts.values, labels=state_counts.index,
    autopct="%1.0f%%", startangle=90,
    colors=[THISTLE, LILAC, SLATE],
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    textprops=dict(color=TEXT, fontsize=9),
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(8.5); at.set_color(BLUSH)
label(ax3, "State breakdown", x=0.5, ha="center")
subtitle(ax3, "Three-state coverage", x=0.5)

# — Urban vs Rural (stacked bar by state)
ax4 = fig.add_subplot(gs[1, 0])
ur = df.groupby(["state","address"]).size().unstack(fill_value=0)
states = ur.index.tolist()
u_vals = ur["U"].values
r_vals = ur["R"].values
x = np.arange(len(states))
ax4.bar(x, u_vals, color=THISTLE, alpha=0.88, label="Urban", width=0.5)
ax4.bar(x, r_vals, bottom=u_vals, color=SLATE, alpha=0.88, label="Rural", width=0.5)
ax4.set_xticks(x); ax4.set_xticklabels(states, fontsize=9)
ax4.legend(loc="upper right")
ax4.yaxis.grid(True); ax4.set_axisbelow(True)
label(ax4, "Urban vs rural by state")
subtitle(ax4, "Near 50/50 split in all three states")

# — Gender split (simple pill bars)
ax5 = fig.add_subplot(gs[1, 1])
sex_counts = df["sex"].value_counts()
colors_sex = [THISTLE, LILAC]
bars5 = ax5.bar(["Female","Male"], sex_counts[["F","M"]].values,
                color=colors_sex, alpha=0.88, width=0.45)
for bar in bars5:
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
             f"{bar.get_height():,}", ha="center", fontsize=9, color=MUTED)
ax5.set_ylim(0, 1900)
ax5.yaxis.grid(True); ax5.set_axisbelow(True)
label(ax5, "Sex breakdown")
subtitle(ax5, "1,548 female · 1,536 male")

# — Internet access + parental involvement
ax6 = fig.add_subplot(gs[1, 2])
pi = df["parental_involvement"].value_counts()[["low","medium","high"]]
inet = df["internet"].value_counts()
categories = ["Low PI","Med PI","High PI","Internet: No","Internet: Yes"]
values = [pi["low"], pi["medium"], pi["high"], inet["no"], inet["yes"]]
colors6 = [SLATE, LILAC, THISTLE, SLATE, THISTLE]
bars6 = ax6.barh(categories, values, color=colors6, alpha=0.85, height=0.55)
ax6.axvline(df.shape[0]/2, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
for bar in bars6:
    ax6.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():,}", va="center", fontsize=8.5, color=MUTED)
ax6.set_xlabel("Students")
ax6.xaxis.grid(True); ax6.set_axisbelow(True)
label(ax6, "Key social factors")
subtitle(ax6, "Parental involvement & internet access")

save(fig, "slide1_demographics.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — OUTCOME HEATMAP BY SUBGROUP
# ══════════════════════════════════════════════════════════════════════════════
print("Slide 2 — outcome heatmap")
fig = plt.figure(figsize=(16, 9), facecolor=BG)
slide_header(fig, "Scanning the full picture",
             "Average outcomes by race × location — cells show deviation from overall mean (z-score)")

ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.18, right=0.88, top=0.85, bottom=0.12)

outcomes = {
    "Avg score":       "avg_score",
    "Attendance (%)":  "attendance_rate",
    "Suspensions":     "suspensions",
    "Failures":        "failures",
}
df["location"] = df["address"].map({"U":"Urban","R":"Rural"})
df["subgroup"] = df["race"] + " · " + df["location"]

rows = []
for sg, grp in df.groupby("subgroup"):
    row = {"Subgroup": sg}
    for label_str, col in outcomes.items():
        row[label_str] = grp[col].mean()
    rows.append(row)

heat_df = pd.DataFrame(rows).set_index("Subgroup")
# z-score each column
heat_z = (heat_df - heat_df.mean()) / heat_df.std()
# flip suspensions and failures so negative = bad
heat_z["Suspensions"] = -heat_z["Suspensions"]
heat_z["Failures"]    = -heat_z["Failures"]

from matplotlib.colors import LinearSegmentedColormap
cmap_custom = LinearSegmentedColormap.from_list(
    "palette", [SLATE, CARD, THISTLE], N=256)

sns.heatmap(heat_z, ax=ax, cmap=cmap_custom,
            center=0, vmin=-2, vmax=2,
            annot=heat_df.round(1), fmt="g",
            annot_kws={"size": 9, "color": BLUSH},
            linewidths=0.8, linecolor=BG,
            cbar_kws={"label": "← worse than avg  |  better than avg →",
                      "shrink": 0.7})
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=9, colors=MUTED)
ax.tick_params(axis="x", labelsize=10, colors=TEXT)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_color(MUTED)
cbar.ax.yaxis.label.set_fontsize(8)
cbar.ax.tick_params(colors=MUTED, labelsize=8)

fig.text(0.88, 0.5, "Suspensions & failures\nare inverted so pink\nalways = better outcome",
         fontsize=8, color=MUTED, va="center", ha="left",
         transform=fig.transFigure)

save(fig, "slide2_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — SCORE GAPS BY LOCATION + AGE
# ══════════════════════════════════════════════════════════════════════════════
print("Slide 3 — score gaps")
fig = plt.figure(figsize=(16, 9), facecolor=BG)
slide_header(fig, "Where the score gaps live",
             "Test scores across race, location, state, and age — each dimension tells a different part of the story")

gs = gridspec.GridSpec(1, 3, figure=fig,
                       left=0.06, right=0.97, top=0.85, bottom=0.10,
                       wspace=0.38)

subjects = ["math_score","reading_score","writing_score"]
sub_labels = ["Math","Reading","Writing"]
sub_colors = [THISTLE, LILAC, SLATE]

# — Panel A: scores by race (grouped bar)
ax1 = fig.add_subplot(gs[0])
race_scores = df.groupby("race")[subjects].mean()
races = race_scores.index.tolist()
x = np.arange(len(races))
w = 0.25
for i,(sub,col) in enumerate(zip(sub_labels, sub_colors)):
    ax1.bar(x + (i-1)*w, race_scores[subjects[i]].values,
            width=w, color=col, alpha=0.88, label=sub)
overall = df["avg_score"].mean()
ax1.axhline(overall, color=BLUSH, linewidth=1.2, linestyle="--",
            alpha=0.6, label=f"Overall mean ({overall:.1f})")
ax1.set_xticks(x)
ax1.set_xticklabels(races, fontsize=8.5, rotation=20, ha="right")
ax1.set_ylabel("Avg score (0–100)")
ax1.set_ylim(0, 75)
ax1.legend(fontsize=8)
ax1.yaxis.grid(True); ax1.set_axisbelow(True)
label(ax1, "A  ·  Scores by race")
subtitle(ax1, "Math / Reading / Writing")

# — Panel B: urban vs rural × state (dot plot)
ax2 = fig.add_subplot(gs[1])
grp = df.groupby(["state","location"])["avg_score"].mean().reset_index()
states = grp["state"].unique()
y_pos = {"Florida":2,"California":1,"Texas":0}
markers = {"Urban":"o","Rural":"s"}
mcolors = {"Urban":THISTLE,"Rural":SLATE}
for _, row in grp.iterrows():
    y = y_pos[row["state"]]
    m = markers[row["location"]]
    c = mcolors[row["location"]]
    ax2.scatter(row["avg_score"], y, marker=m, color=c, s=100, zorder=4)
    ax2.text(row["avg_score"], y+0.12, f"{row['avg_score']:.1f}",
             ha="center", fontsize=8.5, color=c)
# connect urban–rural per state
for state, yd in y_pos.items():
    sub = grp[grp["state"]==state].set_index("location")["avg_score"]
    if "Urban" in sub and "Rural" in sub:
        ax2.plot([sub["Urban"],sub["Rural"]], [yd,yd],
                 color=MUTED, linewidth=1.2, alpha=0.5, zorder=3)
ax2.set_yticks(list(y_pos.values()))
ax2.set_yticklabels(list(y_pos.keys()), fontsize=9)
ax2.set_xlabel("Avg score")
ax2.set_xlim(44, 58)
ax2.xaxis.grid(True); ax2.set_axisbelow(True)
legend_els = [mpatches.Patch(color=THISTLE,label="Urban"),
              mpatches.Patch(color=SLATE,  label="Rural")]
ax2.legend(handles=legend_els, fontsize=8)
label(ax2, "B  ·  State × location")
subtitle(ax2, "Urban vs rural gap by state")

# — Panel C: slope chart — avg score by age
ax3 = fig.add_subplot(gs[2])
age_race = df.groupby(["age","race"])["avg_score"].mean().reset_index()
races_plot = df["race"].unique()
for i, race in enumerate(races_plot):
    sub = age_race[age_race["race"]==race].sort_values("age")
    ax3.plot(sub["age"], sub["avg_score"],
             color=CAT5[i], linewidth=1.8, marker="o",
             markersize=4, label=race, alpha=0.9)
    # label at last point
    last = sub.iloc[-1]
    ax3.text(last["age"]+0.08, last["avg_score"],
             race, fontsize=7.5, color=CAT5[i], va="center")
ax3.set_xticks([15,16,17,18,19])
ax3.set_xlabel("Student age")
ax3.set_ylabel("Avg score")
ax3.set_xlim(14.8, 20.2)
ax3.yaxis.grid(True); ax3.set_axisbelow(True)
label(ax3, "C  ·  Age trajectories")
subtitle(ax3, "Score trend across 15–19 by race")

save(fig, "slide3_score_gaps.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — DISCIPLINE & ATTENDANCE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
print("Slide 4 — discipline & attendance")
fig = plt.figure(figsize=(16, 9), facecolor=BG)
slide_header(fig, "Discipline & attendance — where inequity compounds",
             "Suspension rates, absences, and attendance gaps broken down by race, state, and location")

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.07, right=0.97, top=0.85, bottom=0.09,
                       hspace=0.48, wspace=0.38)

# — Panel A: suspension rate by race (diverging from mean)
ax1 = fig.add_subplot(gs[0, 0])
race_susp = df.groupby("race")["suspensions"].mean().sort_values()
grand_mean = df["suspensions"].mean()
diffs = race_susp - grand_mean
colors_div = [THISTLE if v >= 0 else SLATE for v in diffs.values]
bars = ax1.barh(diffs.index, diffs.values, color=colors_div,
                alpha=0.88, height=0.55)
ax1.axvline(0, color=MUTED, linewidth=1)
for bar, val in zip(bars, diffs.values):
    ax1.text(val + (0.01 if val>=0 else -0.01),
             bar.get_y()+bar.get_height()/2,
             f"{val:+.2f}", va="center",
             ha="left" if val>=0 else "right", fontsize=8.5, color=BLUSH)
ax1.set_xlabel(f"Avg suspensions vs mean ({grand_mean:.2f})")
ax1.xaxis.grid(True); ax1.set_axisbelow(True)
label(ax1, "A  ·  Suspension gap by race")
subtitle(ax1, "Deviation from overall average")

# — Panel B: avg absences by state × location (grouped)
ax2 = fig.add_subplot(gs[0, 1])
abs_grp = df.groupby(["state","location"])["absences"].mean().unstack()
states = abs_grp.index.tolist()
x = np.arange(len(states))
ax2.bar(x-0.18, abs_grp["Urban"].values, width=0.33,
        color=THISTLE, alpha=0.88, label="Urban")
ax2.bar(x+0.18, abs_grp["Rural"].values, width=0.33,
        color=SLATE, alpha=0.88, label="Rural")
ax2.set_xticks(x); ax2.set_xticklabels(states, fontsize=9)
ax2.set_ylabel("Avg absences (days)")
ax2.legend(fontsize=8)
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
label(ax2, "B  ·  Absences by state & location")
subtitle(ax2, "Urban vs rural differences within each state")

# — Panel C: attendance distribution — area by age group
ax3 = fig.add_subplot(gs[1, 0])
bins = np.linspace(60, 100, 28)
centers = (bins[:-1]+bins[1:])/2
age_colors_plot = [THISTLE, SLATE, "#8B7FA8", "#6E7A87", "#59656F"]
for i, age in enumerate([15,16,17,18,19]):
    vals = df[df["age"]==age]["attendance_rate"].values
    hist, _ = np.histogram(vals, bins=bins, density=True)
    ax3.fill_between(centers, hist, alpha=0.25, color=age_colors_plot[i])
    ax3.plot(centers, hist, linewidth=1.4, color=age_colors_plot[i],
             label=str(age))
ax3.set_xlabel("Attendance rate (%)")
ax3.set_ylabel("Density")
ax3.legend(title="Age", fontsize=8, title_fontsize=8)
ax3.yaxis.grid(True); ax3.set_axisbelow(True)
label(ax3, "C  ·  Attendance by age")
subtitle(ax3, "Distribution shape across age cohorts")

# — Panel D: expulsion rate by race × state bubble
ax4 = fig.add_subplot(gs[1, 1])
bubble = df.groupby(["race","state"]).agg(
    susp_mean=("suspensions","mean"),
    att_mean=("attendance_rate","mean"),
    n=("avg_score","count")
).reset_index()
state_markers = {"Florida":"o","California":"s","Texas":"^"}
race_color_map = dict(zip(df["race"].unique(), CAT5))
for _, row in bubble.iterrows():
    ax4.scatter(row["susp_mean"], row["att_mean"],
                s=row["n"]*0.6,
                color=race_color_map[row["race"]],
                marker=state_markers[row["state"]],
                alpha=0.75, edgecolors=CARD, linewidth=0.8, zorder=3)
# legend: race (color) + state (marker)
race_patches = [mpatches.Patch(color=race_color_map[r], label=r)
                for r in df["race"].unique()]
from matplotlib.lines import Line2D
state_lines = [Line2D([0],[0], marker=m, color=MUTED, linestyle="None",
                      markersize=7, label=s)
               for s, m in state_markers.items()]
leg1 = ax4.legend(handles=race_patches, fontsize=7.5,
                  loc="lower left", title="Race", title_fontsize=7.5)
ax4.add_artist(leg1)
ax4.legend(handles=state_lines, fontsize=7.5,
           loc="lower right", title="State", title_fontsize=7.5)
ax4.set_xlabel("Avg suspensions")
ax4.set_ylabel("Avg attendance (%)")
ax4.xaxis.grid(True); ax4.yaxis.grid(True); ax4.set_axisbelow(True)
label(ax4, "D  ·  Suspensions vs attendance")
subtitle(ax4, "Each bubble = race × state group (size = n)")

save(fig, "slide4_discipline.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("Slide 5 — feature importance")
fig = plt.figure(figsize=(16, 9), facecolor=BG)
slide_header(fig, "What actually drives academic outcomes?",
             "Random forest feature importance + Pearson correlations with each score type")

gs = gridspec.GridSpec(1, 2, figure=fig,
                       left=0.06, right=0.97, top=0.85, bottom=0.09,
                       wspace=0.42)

features = ["studytime","failures","absences","Dalc","Walc",
            "famrel","freetime","goout","health",
            "Medu","Fedu","traveltime","age",
            "internet","famsup","schoolsup","paid",
            "activities","higher","romantic",
            "counseling","teacher_support","parental_involvement"]
binary_cols = ["internet","famsup","schoolsup","paid",
               "activities","higher","romantic","counseling","teacher_support"]
ordinal_map = {"low":0,"medium":1,"high":2}

df_enc = df.copy()
for col in binary_cols:
    df_enc[col] = (df_enc[col]=="yes").astype(int)
df_enc["parental_involvement"] = df_enc["parental_involvement"].map(ordinal_map)

X = df_enc[features].fillna(0)
y = df_enc["avg_score"]

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features).sort_values()

human_labels = {
    "studytime":"Study time","failures":"Course failures","absences":"Absences",
    "Dalc":"Weekday alcohol","Walc":"Weekend alcohol","famrel":"Family relations",
    "freetime":"Free time","goout":"Going out","health":"Health",
    "Medu":"Mother's education","Fedu":"Father's education",
    "traveltime":"Travel time","age":"Age","internet":"Internet access",
    "famsup":"Family support","schoolsup":"School support","paid":"Paid tutoring",
    "activities":"Extracurriculars","higher":"Aspires to higher ed",
    "romantic":"Romantic relationship","counseling":"Counseling",
    "teacher_support":"Teacher support","parental_involvement":"Parental involvement",
}

# — Panel A: importance bar
ax1 = fig.add_subplot(gs[0])
q75 = importances.quantile(0.75)
q40 = importances.quantile(0.40)
bar_cols = [THISTLE if v>=q75 else LILAC if v>=q40 else SLATE
            for v in importances.values]
bars = ax1.barh([human_labels.get(f,f) for f in importances.index],
                importances.values, color=bar_cols, alpha=0.88,
                height=0.65)
for bar in bars:
    ax1.text(bar.get_width()+0.0008,
             bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.3f}",
             va="center", ha="left", fontsize=7.5, color=MUTED)
ax1.set_xlabel("Feature importance (mean decrease in impurity)")
ax1.xaxis.grid(True); ax1.set_axisbelow(True)
patches = [mpatches.Patch(color=THISTLE,label="High (top 25%)"),
           mpatches.Patch(color=LILAC,  label="Medium"),
           mpatches.Patch(color=SLATE,  label="Low")]
ax1.legend(handles=patches, fontsize=8, loc="lower right")
label(ax1, "A  ·  Feature importance ranking")
subtitle(ax1, "Predicting average test score")

# — Panel B: correlation heatmap top 10 vs score types
ax2 = fig.add_subplot(gs[1])
top10 = importances.tail(10).index.tolist()
corr_cols = top10 + ["math_score","reading_score","writing_score","attendance_rate"]
corr_m = df_enc[corr_cols].corr().loc[
    top10, ["math_score","reading_score","writing_score","attendance_rate"]
]
corr_m.index = [human_labels.get(f,f) for f in corr_m.index]
corr_m.columns = ["Math","Reading","Writing","Attendance"]

from matplotlib.colors import LinearSegmentedColormap
div_cmap = LinearSegmentedColormap.from_list("div", [LILAC, SLATE, "#e4f49a"], N=256)
sns.heatmap(corr_m, ax=ax2, cmap=div_cmap,
            center=0, vmin=-0.35, vmax=0.35,
            annot=True, fmt=".2f",
            annot_kws={"size":9,"color":BLUSH},
            linewidths=0.6, linecolor=BG,
            cbar_kws={"label":"Pearson r","shrink":0.75})
ax2.tick_params(axis="x", labelsize=9, colors=TEXT)
ax2.tick_params(axis="y", labelsize=8.5, colors=MUTED)
cbar2 = ax2.collections[0].colorbar
cbar2.ax.tick_params(colors=MUTED, labelsize=8)
cbar2.ax.yaxis.label.set_color(MUTED)
label(ax2, "B  ·  Correlation with outcomes")
subtitle(ax2, "Top 10 features vs each outcome type")

save(fig, "slide5_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — INTERVENTION GAP
# ══════════════════════════════════════════════════════════════════════════════
print("Slide 6 — intervention gap")
fig = plt.figure(figsize=(16, 9), facecolor=BG)
slide_header(fig, "The intervention gap — high risk, no support",
             "Students with 3+ failures or 2+ suspensions who receive neither counseling nor teacher support")

gs = gridspec.GridSpec(1, 3, figure=fig,
                       left=0.06, right=0.97, top=0.85, bottom=0.10,
                       wspace=0.40)

df["at_risk"]    = ((df["failures"]>=3)|(df["suspensions"]>=2)).astype(int)
df["no_support"] = ((df["counseling"]=="no")&(df["teacher_support"]=="no")).astype(int)
df["gap"]        = (df["at_risk"] & df["no_support"]).astype(int)

# — Panel A: gap % by race
ax1 = fig.add_subplot(gs[0])
race_gap = df.groupby("race").agg(gap_pct=("gap","mean"),
                                   at_risk_pct=("at_risk","mean")).reset_index()
race_gap = race_gap.sort_values("gap_pct")
x = np.arange(len(race_gap))
ax1.bar(x, race_gap["at_risk_pct"]*100, color=SLATE, alpha=0.55,
        label="At-risk", width=0.5)
ax1.bar(x, race_gap["gap_pct"]*100, color=THISTLE, alpha=0.90,
        label="At-risk + no support", width=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(race_gap["race"], rotation=18, ha="right", fontsize=8.5)
ax1.set_ylabel("% of students")
ax1.legend(fontsize=8)
ax1.yaxis.grid(True); ax1.set_axisbelow(True)
label(ax1, "A  ·  Gap by race")
subtitle(ax1, "Thistle = unsupported at-risk students")

# — Panel B: gap % by state × location
ax2 = fig.add_subplot(gs[1])
sl_gap = df.groupby(["state","location"])["gap"].mean().unstack()*100
states = sl_gap.index.tolist()
x = np.arange(len(states))
ax2.bar(x-0.18, sl_gap["Urban"].values, width=0.33,
        color=THISTLE, alpha=0.88, label="Urban")
ax2.bar(x+0.18, sl_gap["Rural"].values, width=0.33,
        color=SLATE, alpha=0.88, label="Rural")
for i, (u, r) in enumerate(zip(sl_gap["Urban"].values, sl_gap["Rural"].values)):
    ax2.text(i-0.18, u+0.3, f"{u:.0f}%", ha="center", fontsize=8, color=THISTLE)
    ax2.text(i+0.18, r+0.3, f"{r:.0f}%", ha="center", fontsize=8, color=MUTED)
ax2.set_xticks(x); ax2.set_xticklabels(states, fontsize=9)
ax2.set_ylabel("% at-risk & unsupported")
ax2.legend(fontsize=8)
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
label(ax2, "B  ·  Gap by state & location")
subtitle(ax2, "Where are students falling through?")

# — Panel C: support receipt heatmap (failures × suspensions)
ax3 = fig.add_subplot(gs[2])
pivot = df.groupby(["failures","suspensions"])["gap"].mean()*100
pivot_m = pivot.unstack(fill_value=0)
from matplotlib.colors import LinearSegmentedColormap
gap_cmap = LinearSegmentedColormap.from_list("gap",[CARD, LILAC, THISTLE],N=256)
sns.heatmap(pivot_m, ax=ax3, cmap=gap_cmap,
            vmin=0, vmax=60,
            annot=True, fmt=".0f",
            annot_kws={"size":10,"color":BLUSH,"weight":"bold"},
            linewidths=0.6, linecolor=BG,
            cbar_kws={"label":"% unsupported at-risk","shrink":0.75})
ax3.set_xlabel("Suspensions")
ax3.set_ylabel("Failures")
ax3.tick_params(axis="both", labelsize=9, colors=MUTED)
cbar3 = ax3.collections[0].colorbar
cbar3.ax.tick_params(colors=MUTED, labelsize=8)
cbar3.ax.yaxis.label.set_color(MUTED)
label(ax3, "C  ·  Risk profile matrix")
subtitle(ax3, "% with no support at each failure × suspension level")

save(fig, "slide6_intervention_gap.png")

print("\nAll 6 slides complete.")