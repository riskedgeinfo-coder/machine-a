
import streamlit as st
import pandas as pd

# Load the cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_49ja_Draws.csv")
    df = df.sort_values("Draw ID").reset_index(drop=True)
    return df

df = load_data()
draw_id_index = {row["Draw ID"]: idx for idx, row in df.iterrows()}

# Title
st.title("🔮 Bet49 Prediction Engine")
st.markdown("Predicts the 2+ color outcomes for the next 5 draws based on historical similarity.")

# Inputs
draw_id = st.number_input("Enter Draw ID", min_value=1, step=1)
draw_sum = st.number_input("Enter Sum of Draw", min_value=1, step=1)

if st.button("Predict"):
    input_vector = (draw_id, draw_sum)
    df["distance"] = df.apply(lambda row: abs(row["Draw ID"] - input_vector[0]) + abs(row["Sum"] - input_vector[1]), axis=1)
    df["has_future"] = df.index <= (len(df) - 6)
    valid_df = df[df["has_future"] & (df["Draw ID"] != draw_id)]
    top_similar = valid_df.nsmallest(50, "distance")

    future_results = {i: {"Red": 0, "Blue": 0, "Green": 0} for i in range(1, 6)}

    for _, row in top_similar.iterrows():
        base_idx = draw_id_index.get(row["Draw ID"], None)
        if base_idx is not None:
            for offset in range(1, 6):
                future_idx = base_idx + offset
                if future_idx < len(df):
                    future_row = df.iloc[future_idx]
                    color_counts = {"Red": 0, "Blue": 0, "Green": 0}
                    for col in ["N1_color", "N2_color", "N3_color", "N4_color", "N5_color", "N6_color"]:
                        c = future_row[col]
                        if c in color_counts:
                            color_counts[c] += 1
                    for color, count in color_counts.items():
                        if count >= 2:
                            future_results[offset][color] += 1

    st.subheader("📊 Prediction Results")
    st.markdown("**Input Draw ID:** {} | **Sum:** {}".format(draw_id, draw_sum))

    st.markdown("| Draw Ahead | Predicted Color | Confidence (%) | Raw Color Counts |")
    st.markdown("|------------|------------------|----------------|-------------------|")

    for i in range(1, 6):
        counts = future_results[i]
        top_color = max(counts, key=counts.get)
        confidence = (counts[top_color] / 50) * 100
        emoji = "🔴" if top_color == "Red" else "🔵" if top_color == "Blue" else "🟩"
        row = "| t+{} | {} {} | {:.2f}% | Red: {}, Blue: {}, Green: {} |".format(
            i, emoji, top_color, confidence, counts["Red"], counts["Blue"], counts["Green"]
        )
        st.markdown(row)
