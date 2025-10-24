# food_waste_nutrition_optimizer.py
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food Waste → Nutrition Optimizer", layout="wide")

# -----------------------------
# Helper / Model functions
# -----------------------------
def generate_regions(n=12, seed=42):
    """Create synthetic regions with population, production, coordinates."""
    rng = np.random.RandomState(seed)
    regions = []
    for i in range(n):
        pop = int(rng.randint(5000, 150000))  # population
        prod_per_person = rng.uniform(1500, 3000)  # calories produced per person per day (kcal)
        waste_pct = rng.uniform(0.05, 0.35)  # fraction of food wasted
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)  # synthetic coordinates
        regions.append({
            "region": f"Region {i+1}",
            "population": pop,
            "prod_per_person": prod_per_person,
            "waste_pct": waste_pct,
            "x": x, "y": y
        })
    df = pd.DataFrame(regions)
    return df

def compute_surplus_deficit(df, demand_per_person):
    """Compute total demand, production, surplus/deficit (kcal/day)."""
    df = df.copy()
    df["demand_total"] = df["population"] * demand_per_person
    df["production_total"] = df["population"] * df["prod_per_person"]
    df["raw_surplus"] = (df["production_total"] - df["demand_total"]).clip(lower=0.0)
    df["raw_deficit"] = (df["demand_total"] - df["production_total"]).clip(lower=0.0)
    df["waste_available"] = df["production_total"] * df["waste_pct"]
    df["recoverable_potential"] = df[["waste_available", "raw_surplus"]].min(axis=1)
    return df

def distance_matrix(coords):
    coords = np.array(coords)
    n = coords.shape[0]
    dist = np.sqrt(((coords.reshape(n,1,2) - coords.reshape(1,n,2))**2).sum(axis=2))
    return dist

def compute_loss_matrix(dist_mat, loss_per_unit_distance):
    """Simple linear loss capped at 0.99"""
    loss = loss_per_unit_distance * dist_mat
    loss = np.clip(loss, 0, 0.99)
    return loss

def greedy_allocation(df, recovery_rate, loss_per_km):
    """Greedy allocation from donors to receivers."""
    n = df.shape[0]
    coords = df[["x","y"]].values
    dist = distance_matrix(coords)
    loss_mat = compute_loss_matrix(dist, loss_per_km)

    donors_idx = list(df.index[df["recoverable_potential"] > 0])
    receivers_idx = list(df.index[df["raw_deficit"] > 0])

    donor_supply = (df.loc[donors_idx, "recoverable_potential"] * recovery_rate).astype(float).to_dict()
    receiver_need = df.loc[receivers_idx, "raw_deficit"].astype(float).to_dict()
    alloc = np.zeros((n, n), dtype=float)

    receivers_sorted = sorted(receivers_idx, key=lambda i: receiver_need[i], reverse=True)

    for r in receivers_sorted:
        need = receiver_need[r]
        if need <= 0:
            continue
        donor_effective = []
        for d in donors_idx:
            if donor_supply.get(d, 0) <= 0:
                continue
            delivered = donor_supply[d] * (1 - loss_mat[d, r])
            if delivered > 0:
                donor_effective.append((d, delivered, loss_mat[d, r], donor_supply[d]))
        donor_effective.sort(key=lambda x: (1 - x[2]), reverse=True)
        for (d, delivered, loss_frac, supply_left) in donor_effective:
            if need <= 0:
                break
            raw_needed = need / max(1e-9, (1 - loss_frac))
            give_raw = min(supply_left, raw_needed)
            delivered_amount = give_raw * (1 - loss_frac)
            alloc[d, r] += delivered_amount
            donor_supply[d] -= give_raw
            need -= delivered_amount
        receiver_need[r] = need
    return alloc, dist, loss_mat, donor_supply, receiver_need

# -----------------------------
# Streamlit UI & App
# -----------------------------
st.title("Food Waste → Nutrition Optimizer 🍎➡️🥗")
st.write("Simulate redistribution of recoverable food waste (kcal/day) from donor regions to needy regions. "
         "This interpretable greedy allocation model demonstrates how food recovery and logistics can reduce hunger.")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Controls")
    n_regions = st.slider("Number of regions", 6, 20, 12)
    demand_per_person = st.number_input("Calorie demand per person (kcal/day)", value=2100, step=50)
    recovery_rate = st.slider("Recovery efficiency (fraction of waste recoverable)", 0.0, 1.0, 0.5, 0.05)
    loss_per_km = st.slider("Transport loss per distance unit (fraction per 100 units)", 0.0, 0.02, 0.005, 0.0005)
    random_seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)
    run_sim = st.button("Run Simulation")

# Generate synthetic data
df_regions = generate_regions(n_regions, seed=int(random_seed))
df_regions = compute_surplus_deficit(df_regions, demand_per_person)

st.subheader("Regions (Synthetic Data)")
st.dataframe(df_regions[['region','population','prod_per_person','waste_pct','production_total','demand_total','raw_surplus','raw_deficit','recoverable_potential']])

if run_sim or True:
    alloc_matrix, dist_mat, loss_mat, donor_supply_remaining, receiver_need_remaining = greedy_allocation(df_regions, recovery_rate, loss_per_km)

    total_surplus = df_regions["raw_surplus"].sum()
    total_recoverable = df_regions["recoverable_potential"].sum()
    total_recovered = (df_regions["recoverable_potential"] * recovery_rate).sum()
    total_allocated = alloc_matrix.sum()
    total_deficit = df_regions["raw_deficit"].sum()
    coverage_pct = 100.0 * total_allocated / total_deficit if total_deficit > 0 else 100.0

    covered_by_received = alloc_matrix.sum(axis=0)
    df_regions["covered_received"] = covered_by_received
    df_regions["coverage_pct"] = (df_regions["covered_received"] / df_regions["raw_deficit"].replace({0: np.nan}) * 100).fillna(0)

    # Metrics
    st.subheader("Summary Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Surplus (kcal/day)", f"{int(total_surplus):,}")
    c2.metric("Total Recoverable (kcal/day)", f"{int(total_recoverable):,}")
    c3.metric("Total Delivered (kcal/day)", f"{int(total_allocated):,}")
    c1.metric("Total Deficit (kcal/day)", f"{int(total_deficit):,}")
    c2.metric("Recovery Efficiency", f"{recovery_rate*100:.1f}%")
    c3.metric("Deficit Coverage", f"{coverage_pct:.1f}%")

    st.markdown("---")

    # Visualization 1: Coverage per region
    st.subheader("Per-Region Deficit Coverage (%)")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    order = df_regions.sort_values("coverage_pct", ascending=False)["region"]
    sns.barplot(x="region", y="coverage_pct", data=df_regions, order=order, ax=ax1)
    ax1.set_ylabel("Coverage %")
    ax1.set_xlabel("Region")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 105)
    st.pyplot(fig1)

    # Visualization 2: Allocation heatmap
    st.subheader("Allocation Matrix (Donor → Receiver) [Delivered kcal/day]")
    alloc_df = pd.DataFrame(alloc_matrix, index=df_regions["region"], columns=df_regions["region"])
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(alloc_df, ax=ax2, cmap="YlGnBu", linewidths=0.5)
    ax2.set_xlabel("Receiver")
    ax2.set_ylabel("Donor")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Visualization 3: Spatial view with flows
    st.subheader("Spatial Flow Visualization")
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    xs = df_regions["x"].values
    ys = df_regions["y"].values
    pops = df_regions["population"].values

    # ✅ FIXED FOR NUMPY 2.0+
    sizes = 50 + (pops - np.min(pops)) / (np.ptp(pops) + 1e-6) * 300
    ax3.scatter(xs, ys, s=sizes, alpha=0.7)

    for i, row in df_regions.iterrows():
        ax3.text(row["x"]+0.8, row["y"]+0.8, row["region"], fontsize=8)

    max_alloc = alloc_matrix.max() if alloc_matrix.size > 0 else 0
    threshold = max_alloc * 0.05  # show only flows >5% of max

    for i in range(alloc_matrix.shape[0]):
        for j in range(alloc_matrix.shape[1]):
            amt = alloc_matrix[i, j]
            if amt > threshold and i != j:
                x0, y0 = df_regions.loc[i, "x"], df_regions.loc[i, "y"]
                x1, y1 = df_regions.loc[j, "x"], df_regions.loc[j, "y"]
                lw = 0.5 + (np.log1p(amt) / np.log1p(max_alloc + 1e-9)) * 3.5
                ax3.arrow(x0, y0, x1 - x0, y1 - y0, linewidth=lw, alpha=0.6, length_includes_head=True, head_width=1.2)

    ax3.set_xlabel("X Coordinate")
    ax3.set_ylabel("Y Coordinate")
    ax3.set_title("Regions & Major Redistribution Flows")
    st.pyplot(fig3)

    # Allocation Table
    st.subheader("Allocation Table (Non-zero Transfers)")

# Rename indexes before stacking to avoid duplicate column names
    alloc_df.index.name = "Donor"
    alloc_df.columns.name = "Receiver"

    alloc_long = alloc_df.stack().reset_index(name="Delivered_kcal")
    alloc_long = alloc_long[alloc_long["Delivered_kcal"] > 0].sort_values("Delivered_kcal", ascending=False)

    st.dataframe(alloc_long.style.format({"Delivered_kcal": "{:,.0f}"}))


    st.markdown("---")
    st.subheader("Interpretation & Key Insights")
    st.write("""
    - The model redistributes recoverable surplus to cover regional nutrition deficits.
    - Recovery rate and transport loss strongly influence coverage efficiency.
    - The greedy approach is explainable and suitable for policy simulation.
    - Future work: linear programming optimization, perishability models, logistics cost constraints.
    """)

