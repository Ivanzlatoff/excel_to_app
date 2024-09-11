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
from PIL import Image

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
charts = []

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
        title=f"Weight Distribution for {sheet_name}",
        color_discrete_sequence=px.colors.sequential.Plasma  # Specify color sequence
    )

    pie_chart_hawb = px.pie(
        department_counts,  # Use the aggregated data
        values="Number of MAWBs",
        names="SHP_Owning Department ID",
        title=f"HAWB Distribution for {sheet_name}",
        color_discrete_sequence=px.colors.sequential.Plasma  # Specify color sequence
    )

    # Plot the charts
    col1, col2 = st.columns(2)
    col1.plotly_chart(pie_chart_weight)
    col2.plotly_chart(pie_chart_hawb)

    # --- BAR CHARTS ---
    bar_chart_destinations_hawb = px.bar(
        one_hawb_per_mawb.groupby("SHP_House Destination (Air)").size().reset_index(name="Number of HAWBs"),
        x="SHP_House Destination (Air)", y="Number of HAWBs",
        title=f"Number of MAWB's for each destination in {sheet_name}",
        color="Number of HAWBs",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )
    bar_chart_destinations_weight = px.bar(
        one_hawb_per_mawb.groupby("SHP_House Destination (Air)")["SHP_Chargeable Weight"].sum().reset_index(),
        x="SHP_House Destination (Air)", y="SHP_Chargeable Weight",
        title=f"Total chargeable weight of shipments for each destination in {sheet_name}",
        color="SHP_Chargeable Weight",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )

    st.plotly_chart(bar_chart_destinations_hawb)
    st.plotly_chart(bar_chart_destinations_weight)

    # --- TOP 10 DESTINATIONS BAR CHARTS ---
    top_10_destinations_hawb = one_hawb_per_mawb.groupby("SHP_House Destination (Air)").size().nlargest(10).reset_index(name="Number of HAWBs")
    top_10_destinations_weight = one_hawb_per_mawb.groupby("SHP_House Destination (Air)")["SHP_Chargeable Weight"].sum().reset_index().nlargest(10, "SHP_Chargeable Weight")

    bar_chart_top_10_destinations_hawb = px.bar(
        top_10_destinations_hawb,
        x="SHP_House Destination (Air)", y="Number of HAWBs",
        title=f"Top 10 Destinations by Number of MAWB's in {sheet_name}",
        color="Number of HAWBs",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )
    bar_chart_top_10_destinations_weight = px.bar(
        top_10_destinations_weight,
        x="SHP_House Destination (Air)", y="SHP_Chargeable Weight",
        title=f"Top 10 Destinations by Total Chargeable Weight in {sheet_name}",
        color="SHP_Chargeable Weight",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )

    st.plotly_chart(bar_chart_top_10_destinations_hawb)
    st.plotly_chart(bar_chart_top_10_destinations_weight)

    # --- TOP 10 DESTINATIONS BAR CHARTS AGGREGATED ---
    # For Chargeable Weight
    df_weight_by_dest = one_hawb_per_mawb.groupby(["SHP_House Destination (Air)", "SHP_Owning Department ID"])["SHP_Chargeable Weight"].sum().reset_index()

    # Sort by chargeable weight and get top 10 destinations
    top_10_weight_by_dest = df_weight_by_dest.groupby("SHP_House Destination (Air)").sum().nlargest(10, "SHP_Chargeable Weight").reset_index()

    # Filter data for top 10 destinations
    filtered_weight_data = df_weight_by_dest[df_weight_by_dest["SHP_House Destination (Air)"].isin(top_10_weight_by_dest["SHP_House Destination (Air)"])]
    
    # Step 1: Sort the data by SHP_Chargeable Weight in descending order
    filtered_weight_data_sorted = filtered_weight_data.sort_values(by="SHP_Chargeable Weight", ascending=False)

    # Step 2: Modify the "SHP_House Destination (Air)" column to include the chargeable weight in the label
    filtered_weight_data_sorted["Destination"] = (
        filtered_weight_data_sorted["SHP_House Destination (Air)"] + 
        " (" + 
        filtered_weight_data_sorted["SHP_Chargeable Weight"].round(2).astype(str) + 
        " kg)"
    )

    # Step 3: Create the bar chart with the sorted destinations and new labels
    bar_chart_top_10_destinations_weight_aggregated = px.bar(
        filtered_weight_data_sorted,
        x="SHP_Chargeable Weight",
        y="SHP_Owning Department ID",
        color="Destination",  # Use the modified label with weight
        color_discrete_sequence=px.colors.qualitative.Vivid,
        orientation="h",
        height=600,
        title=f"Top 10 Destination by Total Chargeable Weight Aggregated in {sheet_name}"
    )
    
    st.plotly_chart(bar_chart_top_10_destinations_weight_aggregated)

    # For Number of MAWBs
    df_mawb_by_dest = one_hawb_per_mawb.groupby(["SHP_House Destination (Air)", "SHP_Owning Department ID"])["HAWB"].size().reset_index(name="Number of MAWBs")

    # Sort by number of MAWBs and get top 10 destinations
    top_10_mawb_by_dest = df_mawb_by_dest.groupby("SHP_House Destination (Air)").sum().nlargest(10, "Number of MAWBs").reset_index()

    # Filter data for top 10 destinations
    filtered_mawb_data = df_mawb_by_dest[df_mawb_by_dest["SHP_House Destination (Air)"].isin(top_10_mawb_by_dest["SHP_House Destination (Air)"])]
    
    # Step 1: Sort the data by SHP_Chargeable Weight in descending order
    filtered_mawb_data_sorted = filtered_mawb_data.sort_values(by="Number of MAWBs", ascending=False)

    # Step 2: Modify the "SHP_House Destination (Air)" column to include the number of MAWB's in the label
    filtered_mawb_data_sorted["Destination"] = (
        filtered_mawb_data_sorted["SHP_House Destination (Air)"] + " (" 
        + filtered_mawb_data_sorted["Number of MAWBs"].round(2).astype(str)
        + ")"
    )

    # Step 3: Create the bar chart with the sorted destinations and new labels
    bar_chart_top_10_destinations_hawb_aggregated = px.bar(
        filtered_mawb_data_sorted,
        x="Number of MAWBs",
        y="SHP_Owning Department ID",
        color="Destination",  # Use the modified label with weight
        color_discrete_sequence=px.colors.qualitative.Vivid,
        orientation="h",
        height=600,
        title=f"Top 10 Destination by number of MAWB's with 1 HAWB Aggregated in {sheet_name}"
    )
    
    st.plotly_chart(bar_chart_top_10_destinations_hawb_aggregated)

    # --- TREND LINE  ---
    st.markdown(f"## Trend Line for {sheet_name}")

    # Assuming weekly data can be derived from SHP_ETD Date
    filtered_dfs[sheet_name]['Week'] = pd.to_datetime(filtered_df['SHP_ETD Date']).dt.to_period('W').apply(lambda r: r.start_time)
    weekly_data = filtered_df.groupby("Week").agg(
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

    # Add the charts to the list for the PDF report
    charts.append((pie_chart_weight, f"Weight Distribution for {sheet_name}"))
    charts.append((pie_chart_hawb, f"HAWB Distribution for {sheet_name}"))
    charts.append((bar_chart_destinations_hawb, f"Number of MAWB's for each destination in {sheet_name}"))
    charts.append((bar_chart_destinations_weight, f"Total chargeable weight of shipments for each destination in {sheet_name}"))
    charts.append((bar_chart_top_10_destinations_hawb, f"Top 10 Destinations by Number of MAWB's in {sheet_name}"))
    charts.append((bar_chart_top_10_destinations_weight, f"Top 10 Destinations by Total Chargeable Weight in {sheet_name}"))
    charts.append((bar_chart_top_10_destinations_weight_aggregated, f"Top 10 Destination by Total Chargeable Weight Aggregated in {sheet_name}"))
    charts.append((bar_chart_top_10_destinations_hawb_aggregated, f"Top 10 Destination by number of MAWB's Aggregated in {sheet_name}"))
    charts.append((trend_chart_weight, f"Weekly Chargeable Weight with Trend Line for {sheet_name}"))
    charts.append((trend_chart_mawb, f"Weekly Number of MAWB's with Trend Line for {sheet_name}"))

# Print report button
if st.button("Print Report"):
    def save_chart_as_image(fig, filename):
        pio.write_image(fig, filename, format='png')

    def create_pdf_report(charts):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        for chart, title in charts:
            # Create a temporary file for saving the chart image
            temp_filename = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    temp_filename = tmpfile.name
                    save_chart_as_image(chart, temp_filename)
                    
                    # Open the image file using Pillow to ensure it is not locked
                    with Image.open(temp_filename) as img:
                        c.drawString(72, height - 72, title)
                        c.drawImage(temp_filename, 72, height - 600, width=width - 144, preserveAspectRatio=True)
                        c.showPage()
            finally:
                # Ensure the temporary file is deleted
                if temp_filename and os.path.isfile(temp_filename):
                    os.remove(temp_filename)

        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    report = create_pdf_report(charts)
    st.download_button(
        label="Download Report",
        data=report,
        file_name="report.pdf",
        mime="application/pdf"
    )