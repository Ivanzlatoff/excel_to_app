import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os

# Set up Streamlit configuration
st.set_page_config(page_title="B2B Analysis", layout="wide")
st.header("Gateway B2B Analysis")
st.subheader("Analyzing MAWB's with only 1 X HAWB")

# Load Excel file
excel_file = "Gateway HAWB MAWB.xlsm"
sheet_names = pd.ExcelFile(excel_file).sheet_names

# Read all sheets into a dictionary
dfs = {sheet: pd.read_excel(excel_file, sheet_name=sheet, usecols="A:H", header=0).dropna() for sheet in sheet_names}

# --- STREAMLIT SIDEBAR ---
with st.sidebar:
    st.markdown("### Filters for All Sheets")

    all_departments = pd.concat([df["SHP_Owning Department ID"] for df in dfs.values()]).unique().tolist()
    all_chargeable_weights = pd.concat([df["SHP_Chargeable Weight"] for df in dfs.values()]).unique().tolist()
    all_house_destinations = pd.concat([df["SHP_House Destination (Air)"] for df in dfs.values()]).unique().tolist()

    chargeable_weight_selection = st.slider(
        "Chargeable Weight:",
        min_value=min(all_chargeable_weights),
        max_value=max(all_chargeable_weights),
        value=(min(all_chargeable_weights), max(all_chargeable_weights)),
    )

    department_selection = st.multiselect("Department:", all_departments, default=all_departments)
    house_destinations_selection = st.multiselect("House Destination:", all_house_destinations, default=all_house_destinations)

# Filter data for all sheets based on selection
filtered_dfs = {}
for sheet_name, df in dfs.items():
    mask = (
        (df["SHP_Chargeable Weight"].between(*chargeable_weight_selection))
        & (df["SHP_Owning Department ID"].isin(department_selection))
        & (df["SHP_House Destination (Air)"].isin(house_destinations_selection))
    )
    filtered_df = df[mask]
    filtered_dfs[sheet_name] = filtered_df
    number_of_results = filtered_df.shape[0]
    st.markdown(f"*Available Results in {sheet_name}: {number_of_results}*")

    # --- PROCESS MAWB WITH ONLY ONE HAWB ---
    one_hawb_per_mawb = filtered_df.groupby("MAWB").filter(lambda x: len(x) == 1)

    # Aggregate the data to count MAWBs per department
    department_counts = one_hawb_per_mawb.groupby("SHP_Owning Department ID").size().reset_index(name="Number of MAWBs")

    # --- PIE CHARTS ---
    pie_chart_weight = px.pie(
        one_hawb_per_mawb,
        values="SHP_Chargeable Weight",
        names="SHP_Owning Department ID",
        title=f"Weight Distribution for {sheet_name}"
    )

    pie_chart_hawb = px.pie(
        department_counts,  # Use the aggregated data
        values="Number of MAWBs",
        names="SHP_Owning Department ID",
        title=f"HAWB Distribution for {sheet_name}"
    )

    # Plot the charts
    col1, col2 = st.columns(2)
    col1.plotly_chart(pie_chart_weight)
    col2.plotly_chart(pie_chart_hawb)

    # --- BAR CHARTS ---
    bar_chart_destinations_hawb = px.bar(
        one_hawb_per_mawb.groupby("SHP_House Destination (Air)").size().reset_index(name="Number of HAWBs"),
        x="SHP_House Destination (Air)", y="Number of HAWBs",
        title=f"Number of MAWB's for each destination in {sheet_name}"
    )
    bar_chart_destinations_weight = px.bar(
        one_hawb_per_mawb.groupby("SHP_House Destination (Air)")["SHP_Chargeable Weight"].sum().reset_index(),
        x="SHP_House Destination (Air)", y="SHP_Chargeable Weight",
        title=f"Total chargeable weight of shipments for each destination in {sheet_name}"
    )

    st.plotly_chart(bar_chart_destinations_hawb)
    st.plotly_chart(bar_chart_destinations_weight)

    # --- TOP 10 DESTINATIONS BAR CHARTS ---
    top_10_destinations_hawb = one_hawb_per_mawb.groupby("SHP_House Destination (Air)").size().nlargest(10).reset_index(name="Number of HAWBs")
    top_10_destinations_weight = one_hawb_per_mawb.groupby("SHP_House Destination (Air)")["SHP_Chargeable Weight"].sum().reset_index().nlargest(10, "SHP_Chargeable Weight")

    bar_chart_top_10_destinations_hawb = px.bar(
        top_10_destinations_hawb,
        x="SHP_House Destination (Air)", y="Number of HAWBs",
        title=f"Top 10 Destinations by Number of MAWB's in {sheet_name}"
    )
    bar_chart_top_10_destinations_weight = px.bar(
        top_10_destinations_weight,
        x="SHP_House Destination (Air)", y="SHP_Chargeable Weight",
        title=f"Top 10 Destinations by Total Chargeable Weight in {sheet_name}"
    )

    st.plotly_chart(bar_chart_top_10_destinations_hawb)
    st.plotly_chart(bar_chart_top_10_destinations_weight)

# --- TREND LINE FOR THE LAST SHEET ---
last_sheet = sheet_names[-1]
st.markdown(f"## Trend Line for {last_sheet}")

df_last_sheet = filtered_dfs[last_sheet]

# Assuming weekly data can be derived from SHP_ETD Date
df_last_sheet['Week'] = pd.to_datetime(df_last_sheet['SHP_ETD Date']).dt.to_period('W').apply(lambda r: r.start_time)
weekly_data = df_last_sheet.groupby("Week").agg(
    total_weight=pd.NamedAgg(column="SHP_Chargeable Weight", aggfunc="sum"),
    mawb_count=pd.NamedAgg(column="MAWB", aggfunc="count")
).reset_index()

# Convert Week to numerical format for linear regression
weekly_data['Week_num'] = (weekly_data['Week'] - weekly_data['Week'].min()).dt.days

# Linear regression model
def add_trend_line(df, x_col, y_col):
    model = LinearRegression()
    X = df[[x_col]].values.reshape(-1, 1)
    y = df[y_col].values
    model.fit(X, y)
    trend_line = model.predict(X)
    return trend_line

# Line chart for chargeable weight with trend line
trend_chart_weight = go.Figure()
trend_chart_weight.add_trace(go.Scatter(x=weekly_data["Week"], y=weekly_data["total_weight"], mode='markers+lines', name='Total Weight', line=dict(color='blue')))
trend_chart_weight.add_trace(go.Scatter(x=weekly_data["Week"], y=add_trend_line(weekly_data, "Week_num", "total_weight"), mode='lines', name='Trend Line', line=dict(color='red', width=2)))
trend_chart_weight.update_layout(title="Weekly Chargeable Weight with Trend Line")

# Line chart for number of MAWBs with trend line
trend_chart_mawb = go.Figure()
trend_chart_mawb.add_trace(go.Scatter(x=weekly_data["Week"], y=weekly_data["mawb_count"], mode='markers+lines', name='MAWB Count', line=dict(color='red')))
trend_chart_mawb.add_trace(go.Scatter(x=weekly_data["Week"], y=add_trend_line(weekly_data, "Week_num", "mawb_count"), mode='lines', name='Trend Line', line=dict(color='blue', width=2)))
trend_chart_mawb.update_layout(title="Weekly Number of MAWBs with Trend Line")

st.plotly_chart(trend_chart_weight)
st.plotly_chart(trend_chart_mawb)

# Print report button
if st.button("Print Report"):
    def save_chart_as_image(fig, filename):
        pio.write_image(fig, filename, format='png')

    def create_pdf_report(charts):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        for chart, title in charts:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                temp_filename = tmpfile.name
                save_chart_as_image(chart, temp_filename)
                c.drawString(72, height - 72, title)
                c.drawImage(temp_filename, 72, height - 600, width=width - 144, preserveAspectRatio=True)
                c.showPage()
                os.remove(temp_filename)

        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    charts = [
        (pie_chart_weight, "Weight Distribution"),
        (pie_chart_hawb, "HAWB Distribution"),
        (bar_chart_destinations_hawb, "Destinations HAWB"),
        (bar_chart_destinations_weight, "Destinations Weight"),
        (bar_chart_top_10_destinations_hawb, "Top 10 Destinations HAWB"),
        (bar_chart_top_10_destinations_weight, "Top 10 Destinations Weight"),
        (trend_chart_weight, "Weekly Chargeable Weight Trend Line"),
        (trend_chart_mawb, "Weekly MAWB Count Trend Line"),
    ]

    report = create_pdf_report(charts)
    st.download_button(
        label="Download Report",
        data=report,
        file_name="report.pdf",
        mime="application/pdf"
    )
