"""
Hacker News Posts Analysis
===========================
Analyzes Ask HN, Show HN, and Other posts from the Hacker News dataset.

Key Questions:
  1. Do Ask HN or Show HN posts receive more comments on average?
  2. At what hour do Ask HN posts receive the most comments?

Dataset columns: Id, Title, URL, Points, Comments, Author, Created_at

Usage:
    python hacker_news_analysis.py

Requirements:
    pip install pandas matplotlib seaborn numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"Ask HN": "#4C72B0", "Show HN": "#DD8452", "Other": "#55A868"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str = "hacker_news.csv") -> pd.DataFrame:
    """Load and clean the Hacker News dataset."""
    df = pd.read_csv(filepath)
    df.columns = ["Id", "Title", "URL", "Points", "Comments", "Author", "Created_at"]
    df["Created_at"] = pd.to_datetime(df["Created_at"])
    df["URL"] = df["URL"].fillna("No URL provided")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. POST CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Post_Type' column labelling each row as
    'Ask HN', 'Show HN', or 'Other'.
    """
    title_lower = df["Title"].str.lower()
    conditions = [
        title_lower.str.startswith("ask hn"),
        title_lower.str.startswith("show hn"),
    ]
    choices = ["Ask HN", "Show HN"]
    df["Post_Type"] = np.select(conditions, choices, default="Other")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def avg_comments_by_type(df: pd.DataFrame) -> pd.Series:
    """Return mean comment count grouped by Post_Type."""
    return df.groupby("Post_Type")["Comments"].mean().reindex(
        ["Ask HN", "Show HN", "Other"]
    )


def avg_comments_by_hour(df: pd.DataFrame) -> pd.Series:
    """
    For Ask HN posts only, return mean comments grouped by
    the hour the post was created (0–23).
    """
    ask_df = df[df["Post_Type"] == "Ask HN"].copy()
    ask_df["Hour"] = ask_df["Created_at"].dt.hour
    return ask_df.groupby("Hour")["Comments"].mean().sort_index()


def monthly_post_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of monthly post counts by Post_Type."""
    return (
        df.set_index("Created_at")
        .groupby("Post_Type")
        .resample("ME")
        .size()
        .reset_index(name="Count")
    )


def top_hours(df: pd.DataFrame, n: int = 5) -> None:
    """Print the top N Ask HN hours by average comments."""
    hourly = avg_comments_by_hour(df).sort_values(ascending=False)
    print(f"\nTop {n} hours for Ask HN comments:")
    print("-" * 38)
    for hour, avg in hourly.head(n).items():
        print(f"  {hour:>2}:00  →  {avg:5.2f} avg comments per post")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_avg_comments_by_type(df: pd.DataFrame) -> None:
    """
    Bar chart: average number of comments for each post type
    (Ask HN / Show HN / Other).
    """
    avgs = avg_comments_by_type(df)
    colors = [PALETTE[t] for t in avgs.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(avgs.index, avgs.values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    # Value labels on top of each bar
    for bar, val in zip(bars, avgs.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Average Comments by Post Type", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Post Type", fontsize=12)
    ax.set_ylabel("Average Comments", fontsize=12)
    ax.set_ylim(0, avgs.max() * 1.25)
    sns.despine()
    plt.tight_layout()
    plt.savefig("plot1_avg_comments_by_type.png", dpi=150)
    plt.show()


def plot_avg_comments_by_hour(df: pd.DataFrame) -> None:
    """
    Bar chart: average Ask HN comments per post for every hour of the day.
    The peak hour (15:00) is highlighted in red.
    """
    hourly = avg_comments_by_hour(df)
    peak_hour = hourly.idxmax()
    bar_colors = ["#e05c5c" if h == peak_hour else "#4C72B0"
                  for h in hourly.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(hourly.index, hourly.values, color=bar_colors,
           edgecolor="white", linewidth=0.8)

    # Annotate the peak bar
    ax.annotate(
        f"Peak: {peak_hour}:00\n({hourly[peak_hour]:.1f} avg)",
        xy=(peak_hour, hourly[peak_hour]),
        xytext=(peak_hour + 2, hourly[peak_hour] * 0.92),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10
    )

    ax.set_title("Average Ask HN Comments by Hour of Day",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hour of Day (24 h)", fontsize=12)
    ax.set_ylabel("Avg Comments per Post", fontsize=12)
    ax.set_xticks(range(0, 24))
    sns.despine()
    plt.tight_layout()
    plt.savefig("plot2_avg_comments_by_hour.png", dpi=150)
    plt.show()


def plot_time_series(df: pd.DataFrame) -> None:
    """
    Line chart: monthly post volume over time, one line per post type.
    """
    monthly = monthly_post_counts(df)

    fig, ax = plt.subplots(figsize=(13, 5))
    for post_type, group in monthly.groupby("Post_Type"):
        ax.plot(
            group["Created_at"], group["Count"],
            marker="o", markersize=4, linewidth=1.8,
            label=post_type, color=PALETTE[post_type]
        )

    ax.set_title("Monthly Post Volume Over Time by Post Type",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.legend(title="Post Type", frameon=True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    fig.autofmt_xdate()
    sns.despine()
    plt.tight_layout()
    plt.savefig("plot3_time_series.png", dpi=150)
    plt.show()


def plot_points_vs_comments(df: pd.DataFrame) -> None:
    """
    Scatter plot: Points vs Comments, colour-coded by post type.
    Axes are capped at the 99th percentile to suppress extreme outliers.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for post_type, group in df.groupby("Post_Type"):
        ax.scatter(
            group["Points"], group["Comments"],
            alpha=0.35, s=18,
            label=post_type, color=PALETTE[post_type]
        )

    ax.set_title("Points vs Comments by Post Type",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Points", fontsize=12)
    ax.set_ylabel("Comments", fontsize=12)
    ax.set_xlim(0, df["Points"].quantile(0.99))
    ax.set_ylim(0, df["Comments"].quantile(0.99))
    ax.legend(title="Post Type", markerscale=2, frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig("plot4_points_vs_comments.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load & prepare ────────────────────────────────────────────────────────
    df = load_data("hacker_news.csv")
    df = classify_posts(df)

    # ── Console summary ───────────────────────────────────────────────────────
    counts = df["Post_Type"].value_counts()
    print("\nPost type distribution:")
    print("-" * 30)
    for ptype, count in counts.items():
        print(f"  {ptype:<10} {count:>6} posts")

    avgs = avg_comments_by_type(df)
    print("\nAverage comments per post type:")
    print("-" * 30)
    for ptype, avg in avgs.items():
        print(f"  {ptype:<10} {avg:>6.2f} avg comments")

    top_hours(df, n=5)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_avg_comments_by_type(df)
    plot_avg_comments_by_hour(df)
    plot_time_series(df)
    plot_points_vs_comments(df)
    print("Done. Plots saved as PNG files in the current directory.")


if __name__ == "__main__":
    main()
